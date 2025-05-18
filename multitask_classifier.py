'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_response_data
from datasets import ResponseSuggestionDataset

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO Mujtaba

        ## Sentiment classifier: projects [CLS] embedding to 5 classes.
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, 5)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        ## For paired tasks, we combine two [CLS] embeddings and their absolute difference.
        ## The combined vector has dimension: hidden_size + hidden_size + hidden_size = 3 * hidden_size.
        self.paraphrase_classifier = nn.Linear(3 * BERT_HIDDEN_SIZE, 1)
        self.similarity_regressor = nn.Linear(3 * BERT_HIDDEN_SIZE, 1)

        self.message_proj = nn.Linear(BERT_HIDDEN_SIZE, 
                                 config.response_suggestion_params['embedding_dim'])
        self.response_proj = nn.Linear(BERT_HIDDEN_SIZE, 
                                    config.response_suggestion_params['embedding_dim'])
        self.response_bias = nn.Parameter(torch.zeros(1)) if \
            config.response_suggestion_params['response_bias'] else None


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO Mujtaba

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs['pooler_output']
        return cls_embedding


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO Mujtaba
        
        cls_embedding = self.forward(input_ids, attention_mask)
        cls_output = self.dropout(cls_embedding)  # Apply dropout
        logits = self.sentiment_classifier(cls_output)
        return logits


    ### TODO Mujtaba
    def _combine_pair(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        '''
        Helper method for paired inputs.
        It computes the [CLS] embedding for each sentence and concatenates:
        [embedding_1; embedding_2; |embedding_1 - embedding_2|]
        '''
        h1 = self.forward(input_ids_1, attention_mask_1)
        h2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=1)
        return combined


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO Mujtaba

        combined = self._combine_pair(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        logit = self.paraphrase_classifier(combined)
        return logit.squeeze(-1)  ## Remove singleton dimension


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO Mujtaba
        
        combined = self._combine_pair(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        logit = self.similarity_regressor(combined)
        return logit.squeeze(-1)
    

    def encode_message(self, input_ids, attention_mask):
        """Encode messages into fixed-size vectors"""
        outputs = self.bert(input_ids, attention_mask)
        cls_embedding = outputs['pooler_output']
        return self.message_proj(cls_embedding)

    def encode_response(self, input_ids, attention_mask):
        """Encode responses into fixed-size vectors"""
        outputs = self.bert(input_ids, attention_mask)
        cls_embedding = outputs['pooler_output']
        return self.response_proj(cls_embedding)

    def score_pair(self, message_emb, response_emb):
        """Dot-product scoring with optional bias"""
        return torch.matmul(message_emb, response_emb.t()) + self.response_bias




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


# def train_multitask(args):
#     device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
#     # Load data for all tasks
#     sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
#         args.sst_train, args.para_train, args.sts_train, split='train'
#     )
#     sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
#         args.sst_dev, args.para_dev, args.sts_dev, split='dev'
#     )
    
#     # Create datasets
#     sst_train_data = SentenceClassificationDataset(sst_train_data, args)
#     sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
#     para_train_data = SentencePairDataset(para_train_data, args)
#     para_dev_data = SentencePairDataset(para_dev_data, args)
#     sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
#     sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    
#     # Create dataloaders
#     sst_train_dataloader = DataLoader(
#         sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn
#     )
#     sst_dev_dataloader = DataLoader(
#         sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn
#     )
#     para_train_dataloader = DataLoader(
#         para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn
#     )
#     para_dev_dataloader = DataLoader(
#         para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn
#     )
#     sts_train_dataloader = DataLoader(
#         sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn
#     )
#     sts_dev_dataloader = DataLoader(
#         sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn
#     )
    
#     # Initialize model
#     config = {
#         'hidden_dropout_prob': args.hidden_dropout_prob,
#         'num_labels': num_labels,
#         'hidden_size': 768,
#         'data_dir': '.',
#         'fine_tune_mode': args.fine_tune_mode
#     }
#     config = SimpleNamespace(**config)
#     model = MultitaskBERT(config)
#     model = model.to(device)
    
#     # Optimizer
#     optimizer = AdamW(model.parameters(), lr=args.lr)
    
#     # Evaluate pre-trained model before training
#     model.eval()
#     with torch.no_grad():
#         dev_sentiment_accuracy, _, _, dev_paraphrase_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(
#             sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
#         )
#         print("Pre-trained model performance:")
#         print(f"Dev sentiment accuracy: {dev_sentiment_accuracy:.3f}")
#         print(f"Dev paraphrase accuracy: {dev_paraphrase_accuracy:.3f}")
#         print(f"Dev STS correlation: {dev_sts_corr:.3f}")
    
#     best_dev_acc = 0
#     sst_losses = []
#     para_losses = []
#     sts_losses = []
#     # Outer scope
#     best_sst_acc = 0
#     best_para_acc = 0
#     best_sts_corr = 0
    
#     # Training function for a single task
#     def train_task(dataloader, task_name, epochs, predict_fn, loss_fn, is_regression=False):
#       task_losses = []
#       for epoch in range(epochs):
#           model.train()
#           train_loss = 0
#           num_batches = 0
#           for batch in tqdm(dataloader, desc=f'{task_name}-train-{epoch}', disable=TQDM_DISABLE):
#               optimizer.zero_grad()
#               if task_name == 'sst':
#                   b_ids = batch['token_ids'].to(device)
#                   b_mask = batch['attention_mask'].to(device)
#                   b_labels = batch['labels'].to(device)
#                   logits = predict_fn(b_ids, b_mask)
#                   loss = loss_fn(logits, b_labels.view(-1), reduction='mean')
#               else:
#                   b_ids1 = batch['token_ids_1'].to(device)
#                   b_mask1 = batch['attention_mask_1'].to(device)
#                   b_ids2 = batch['token_ids_2'].to(device)
#                   b_mask2 = batch['attention_mask_2'].to(device)
#                   b_labels = batch['labels'].to(device)
#                   logits = predict_fn(b_ids1, b_mask1, b_ids2, b_mask2)
#                   if is_regression:
#                       # Convert b_labels to float32 for STS regression
#                       b_labels = b_labels.float()  # Add this line
#                       loss = loss_fn(logits, b_labels, reduction='mean')
#                   else:
#                       # For paraphrase, ensure labels are float32
#                       loss = loss_fn(logits, b_labels.float(), reduction='mean')
              
#               loss.backward()
#               optimizer.step()
#               train_loss += loss.item()
#               num_batches += 1
          
#           train_loss /= num_batches
#           task_losses.append(train_loss)
#           model.eval()
#           with torch.no_grad():
#               dev_sentiment_accuracy, _, _, dev_paraphrase_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(
#                   sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
#               )
#           print(f"Epoch {epoch+1} ({task_name}): Train {task_name} loss: {train_loss:.3f}, "
#                 f"Dev sentiment: {dev_sentiment_accuracy:.3f}, Dev paraphrase: {dev_paraphrase_accuracy:.3f}, "
#                 f"Dev STS: {dev_sts_corr:.3f}")
#           nonlocal best_dev_acc

#           # Inside train_task
#           nonlocal best_sts_corr
#           nonlocal best_sst_acc
#           nonlocal best_para_acc
#           improved = False
#           if dev_sentiment_accuracy > best_sst_acc:
#               best_sst_acc = dev_sentiment_accuracy
#               improved = True
#           if dev_paraphrase_accuracy > best_para_acc:
#               best_para_acc = dev_paraphrase_accuracy
#               improved = True
#           if dev_sts_corr > best_sts_corr:
#               best_sts_corr = dev_sts_corr
#               improved = True
#           if improved:
#               save_model(model, optimizer, args, config, args.filepath)
#       return task_losses
    
#     # Train on Paraphrase
#     print("\nTraining on Paraphrase...")
#     para_losses = train_task(
#         para_train_dataloader, 'para', 1, 
#         model.predict_paraphrase, F.binary_cross_entropy_with_logits
#     )

#     # Train on STS
#     print("\nTraining on STS...")
#     sts_losses = train_task(
#         sts_train_dataloader, 'sts', args.epochs*5, 
#         model.predict_similarity, F.mse_loss, is_regression=True
#     )
    
#     # Train on SST
#     print("\nTraining on SST...")
#     sst_losses = train_task(
#         sst_train_dataloader, 'sst', (args.epochs*2)+1, 
#         model.predict_sentiment, F.cross_entropy
#     )
    
#     # Plot training losses for each task
#     plt.figure(figsize=(12, 8))
#     plt.plot(range((args.epochs*2)+1), sst_losses, label='SST Loss', marker='o')
#     plt.plot(range(1), para_losses, label='Paraphrase Loss', marker='o')
#     plt.plot(range(args.epochs*5), sts_losses, label='STS Loss', marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel('Training Loss')
#     plt.title('Training Loss Over Epochs for Each Task (Sequential Training)')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('sequential_training_loss_plot.png')
#     plt.show()
#     print("Training loss plot saved as 'sequential_training_loss_plot.png'")

def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load data, datasets, and dataloaders
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')
    
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)

    rs_train_data = load_response_data(args.para_train)
    rs_dev_data = load_response_data(args.para_dev)  
    rs_train_dataset = ResponseSuggestionDataset(rs_train_data, args)
    rs_dev_dataset = ResponseSuggestionDataset(rs_dev_data, args)


    # Initialize model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob, 'num_labels': num_labels, 'hidden_size': 768, 'data_dir': '.', 'fine_tune_mode': args.fine_tune_mode,
    'response_suggestion_params': {
        'embedding_dim': args.embedding_dim,
        'response_bias': args.response_bias,
        'num_responses': 10000
    }}
    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Evaluate pre-trained model
    model.eval()
    with torch.no_grad():
        dev_sentiment_accuracy, _, _, dev_paraphrase_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
        )
        print("Pre-trained model performance:")
        print(f"Dev sentiment accuracy: {dev_sentiment_accuracy:.3f}")
        print(f"Dev paraphrase accuracy: {dev_paraphrase_accuracy:.3f}")
        print(f"Dev STS correlation: {dev_sts_corr:.3f}")
    
    best_dev_acc = 0
    train_total_losses = []
    # best_sst_acc = 0
    # best_para_acc = 0
    # best_sts_corr = 0
    
    # Calculate steps per task based on dataset size and desired "epoch equivalent"
    sst_steps = len(sst_train_dataloader)  # Full epoch of SST
    para_steps = int(len(para_train_dataloader) * 0.2)  # 20% of paraphrase epoch (reduce time)
    sts_steps = len(sts_train_dataloader)  # Full epoch of STS
    num_steps = max(sst_steps, para_steps, sts_steps)
    
    for epoch in range(args.epochs):
        model.train()
        train_sst_loss = 0
        train_para_loss = 0
        train_sts_loss = 0
        num_batches = 0
        
        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)
        
        for step in tqdm(range(num_steps), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            total_loss = 0
            # SST
            if step < sst_steps:
                try:
                    sst_batch = next(sst_iter)
                except StopIteration:
                    sst_iter = iter(sst_train_dataloader)
                    sst_batch = next(sst_iter)
                sst_ids = sst_batch['token_ids'].to(device)
                sst_mask = sst_batch['attention_mask'].to(device)
                sst_labels = sst_batch['labels'].to(device).long()  # Ensure labels are torch.long for classification
                sst_logits = model.predict_sentiment(sst_ids, sst_mask)
                sst_loss = F.cross_entropy(sst_logits, sst_labels.view(-1), reduction='mean')
                total_loss += sst_loss
                train_sst_loss += sst_loss.item()
            else:
                sst_loss = 0
            
            # Paraphrase
            if step < para_steps:
                try:
                    para_batch = next(para_iter)
                except StopIteration:
                    para_iter = iter(para_train_dataloader)
                    para_batch = next(para_iter)
                para_ids1 = para_batch['token_ids_1'].to(device)
                para_mask1 = para_batch['attention_mask_1'].to(device)
                para_ids2 = para_batch['token_ids_2'].to(device)
                para_mask2 = para_batch['attention_mask_2'].to(device)
                para_labels = para_batch['labels'].to(device).float()  # Ensure labels are torch.float32 for BCE
                para_logits = model.predict_paraphrase(para_ids1, para_mask1, para_ids2, para_mask2)
                para_loss = F.binary_cross_entropy_with_logits(para_logits, para_labels.float(), reduction='mean')
                total_loss += para_loss
                train_para_loss += para_loss.item()
            else:
                para_loss = 0
            
            # STS
            if step < sts_steps:
                try:
                    sts_batch = next(sts_iter)
                except StopIteration:
                    sts_iter = iter(sts_train_dataloader)
                    sts_batch = next(sts_iter)
                sts_ids1 = sts_batch['token_ids_1'].to(device)
                sts_mask1 = sts_batch['attention_mask_1'].to(device)
                sts_ids2 = sts_batch['token_ids_2'].to(device)
                sts_mask2 = sts_batch['attention_mask_2'].to(device)
                sts_labels = sts_batch['labels'].to(device).float()  # Ensure labels are torch.float32 for MSE
                sts_pred = model.predict_similarity(sts_ids1, sts_mask1, sts_ids2, sts_mask2)
                sts_loss = F.mse_loss(sts_pred, sts_labels, reduction='mean')
                total_loss += sts_loss
                train_sts_loss += sts_loss.item()
            else:
                sts_loss = 0
            
            if total_loss > 0:  # Only optimize if at least one task contributed
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                num_batches += 1
        
        # Average losses
        train_sst_loss = train_sst_loss / max(1, min(sst_steps, num_steps))
        train_para_loss = train_para_loss / max(1, min(para_steps, num_steps))
        train_sts_loss = train_sts_loss / max(1, min(sts_steps, num_steps))
        train_total_loss = train_sst_loss + train_para_loss + train_sts_loss
        train_total_losses.append(train_total_loss)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            dev_sentiment_accuracy, _, _, dev_paraphrase_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(
                sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
            )
        print(f"Epoch {epoch}: Train SST loss: {train_sst_loss:.3f}, Para loss: {train_para_loss:.3f}, "
              f"STS loss: {train_sts_loss:.3f}, Total loss: {train_total_loss:.3f}")
        print(f"Dev sentiment: {dev_sentiment_accuracy:.3f}, Dev paraphrase: {dev_paraphrase_accuracy:.3f}, "
              f"Dev STS: {dev_sts_corr:.3f}")
        # if dev_sentiment_accuracy > best_sst_acc:
        #       best_sst_acc = dev_sentiment_accuracy
        #       improved = True
        # if dev_paraphrase_accuracy > best_para_acc:
        #     best_para_acc = dev_paraphrase_accuracy
        #     improved = True
        # if dev_sts_corr > best_sts_corr:
        #     best_sts_corr = dev_sts_corr
        #     improved = True
        # if improved:
        #     save_model(model, optimizer, args, config, args.filepath)
        if dev_sentiment_accuracy > best_dev_acc:
            best_dev_acc = dev_sentiment_accuracy
            save_model(model, optimizer, args, config, args.filepath)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), train_total_losses, label='Total Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Total Training Loss')
    plt.title('Reduction of Training Loss Over Epochs (Simultaneous)')
    plt.legend()
    plt.grid(True)
    plt.savefig('simultaneous_training_loss_plot.png')
    plt.show()

    # After the training loop in train_multitask()
    if hasattr(config, 'response_suggestion_params'):
        rs_acc = evaluate_response_suggestion(model, rs_dev_dataloader, device)
        print(f"Dev Response Suggestion Accuracy: {rs_acc:.3f}")




def train_response_suggestion(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='training'):
        msg_ids = batch['msg_ids'].to(device)
        msg_mask = batch['msg_mask'].to(device)
        resp_ids = batch['resp_ids'].to(device)
        resp_mask = batch['resp_mask'].to(device)
        
        # Encode all messages and responses
        msg_embs = model.encode_message(msg_ids, msg_mask)
        resp_embs = model.encode_response(resp_ids, resp_mask)
        
        # Compute scores matrix (batch_size x batch_size)
        scores = model.score_pair(msg_embs, resp_embs)
        
        # Diagonal elements are positive pairs
        targets = torch.arange(len(msg_embs)).to(device)
        
        # Softmax cross-entropy loss
        loss = F.cross_entropy(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches




def evaluate_response_suggestion(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            msg_ids = batch['msg_ids'].to(device)
            msg_mask = batch['msg_mask'].to(device)
            resp_ids = batch['resp_ids'].to(device)
            resp_mask = batch['resp_mask'].to(device)
            
            msg_embs = model.encode_message(msg_ids, msg_mask)
            resp_embs = model.encode_response(resp_ids, resp_mask)
            
            scores = model.score_pair(msg_embs, resp_embs)
            preds = scores.argmax(dim=1)
            correct += (preds == torch.arange(len(msg_embs)).to(device)).sum().item()
            total += len(msg_embs)
    
    accuracy = correct / total
    return accuracy




def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    # Add these with the other dataset arguments
    parser.add_argument("--rs_train", type=str, default="data/response-train.csv",help="Path to response suggestion training data")
    parser.add_argument("--rs_dev", type=str, default="data/response-dev.csv",
                    help="Path to response suggestion dev data")

    # Add these with the other output file arguments
    parser.add_argument("--rs_dev_out", type=str, default="predictions/rs-dev-output.csv",help="Output file for response suggestion dev predictions")
    parser.add_argument("--rs_test_out", type=str, default="predictions/rs-test-output.csv",help="Output file for response suggestion test predictions")

    # Add response suggestion specific parameters
    parser.add_argument("--embedding_dim", type=int, default=256,help="Dimension for response suggestion embeddings")
    parser.add_argument("--response_bias", action='store_true',help="Whether to use bias term in response scoring")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-{args.batch_size}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
