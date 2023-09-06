from datetime import datetime
import logging
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from google.cloud import bigquery
from transformers import CamembertTokenizer, CamembertForSequenceClassification, get_linear_schedule_with_warmup

# Fonctions utilitaires
def tokenize_text(text, tokenizer):
    return tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
def batch_insert_rows(client, table_id, rows, batch_size=1000):
    # Split rows into batches of size 'batch_size'
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    for idx, batch in enumerate(batches, 1):
        logging.info(
            f"Inserting batch {idx} of {len(batches)} with {len(batch)} rows.")

        errors = client.insert_rows_json(table_id, batch)
        if not errors:
            logging.info(f"Batch {idx} successfully inserted.")
        else:
            logging.error(
                f"Errors encountered during insertion of batch {idx}: {errors}")

# Initialisation
logging.basicConfig(level=logging.INFO)
now = datetime.now()
print(f"Starting time: {now}")

# Connexion à BigQuery
credential_path = "C:/Users/aiche/PycharmProjects/pythonProject/clejson.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
client = bigquery.Client()

# Requêtes
QUERY_test = "SELECT texte,label-1 as label from production_saqara_data.o_categories_lots_tests"
QUERY_template = "SELECT distinct trade_name from production_saqara_data.i_aos_templates where trade_name not in (select trade_name from production_saqara_data.test_categorie_bart) limit 5000"
QUERY_categorie = "SELECT id-1 as id,categorie_3 from production_saqara_data.o_list_categories_lots"

# Récupération des données
df_test = client.query(QUERY_test).to_dataframe()
df_to_categorize = client.query(QUERY_template).to_dataframe()
df_categories = client.query(QUERY_categorie).to_dataframe()

# Tokenization
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
df_test['tokenized'] = df_test['texte'].apply(lambda x: tokenize_text(x, tokenizer))
df_to_categorize['tokenized'] = df_to_categorize['trade_name'].apply(lambda x: tokenize_text(x, tokenizer))

# Préparation des données
labels = torch.tensor(df_test['label'].values)
input_ids = pad_sequence([t['input_ids'].squeeze(0) for t in df_test['tokenized']], batch_first=True)
attention_masks = pad_sequence([t['attention_mask'].squeeze(0) for t in df_test['tokenized']], batch_first=True)

# Division et DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size, val_size = int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 8
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Modèle
model = CamembertForSequenceClassification.from_pretrained(
    'camembert-base',
    num_labels=len(df_test['label'].unique()),
    output_attentions=False,
    output_hidden_states=False
)

# Entraînement et évaluation
optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)


# Boucle d'entraînement
for epoch in range(epochs):
    # Mode d'entraînement
    model.train()

    for batch in train_dataloader:
        b_input_ids, b_attention_mask, b_labels = batch

        # Effacer les gradients précédents
        model.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()

        # Mise à jour des paramètres
        optimizer.step()
        scheduler.step()

    # Mode d'évaluation
    model.eval()

    # Variables pour les métriques d'évaluation
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Évaluation
    for batch in validation_dataloader:
        b_input_ids, b_attention_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)

        # Calcul du loss
        loss = outputs.loss
        logits = outputs.logits


        # Accumuler le loss total
        total_eval_loss += loss.item()

        # Déplacer logits et labels vers CPU pour le calcul des métriques
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculez l'exactitude pour ce batch et accumulez-le sur tous les batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    # Rapport final
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    print("Précision en validation: {0:.2f}".format(avg_val_accuracy))
    print("Perte en validation: {0:.2f}".format(avg_val_loss))

# Sauvegarde du modèle
model.save_pretrained("modele/camembert")

print("model save")

#model = CamembertForSequenceClassification.from_pretrained("modele/camembert")
input_ids_list = [t['input_ids'].squeeze(0) for t in df_to_categorize['tokenized']]
attention_mask_list = [t['attention_mask'].squeeze(0) for t in df_to_categorize['tokenized']]


# Utilisez pad_sequence pour rendre les deux listes de la même longueur
padded_input_ids = pad_sequence(input_ids_list, batch_first=True)
padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True)

# Utilisation du modèle
with torch.no_grad():
    outputs = model(padded_input_ids, attention_mask=padded_attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)

max_probabilities, _ = torch.max(probabilities, dim=1)
max_probabilities = max_probabilities.tolist()

# Ajout des probabilités max au DataFrame
label_map = {row['id']: row['categorie_3'] for index, row in df_categories.iterrows()}

# Obtention des prédictions
predictions = torch.argmax(logits, dim=1)
readable_labels = [label_map[label.item()] for label in predictions]

predicted_ids = [label.item() for label in predictions]
df_to_categorize['predicted_label_id'] = predicted_ids

# Ajout des étiquettes aux données originales
df_to_categorize['predicted_label'] = readable_labels
df_to_categorize['max_probability'] = max_probabilities

rows_to_insert = []
table_id = 'production_saqara_data.test_categorie_bart'

df_to_categorize['input_ids'] = df_to_categorize['tokenized'].apply(lambda x: x['input_ids'][0].tolist())
df_to_categorize['attention_mask'] = df_to_categorize['tokenized'].apply(lambda x: x['attention_mask'][0].tolist())

# Supprimez la colonne tokenized si elle n'est plus nécessaire
del df_to_categorize['tokenized']
del df_to_categorize['input_ids']
del df_to_categorize['attention_mask']

now = datetime.now()
print("now after model =", now)

# Insérez les lignes dans la table
rows_to_insert = df_to_categorize.to_dict('records')
batch_insert_rows(client, table_id, rows_to_insert, batch_size=1000)

now = datetime.now()
print("now finish =", now)