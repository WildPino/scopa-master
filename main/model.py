import torch
import torch.nn as nn
import torch.nn.functional as F

class ScopaNet(nn.Module):
    def __init__(self, input_dim=255, action_dim=41):
        super(ScopaNet, self).__init__()
        
        # 1. TRONCO COMUNE (Shared Backbone)
        # La rete analizza lo stato del gioco. Questi layer sono usati sia
        # dall'Attore che dal Critico per capire "cosa sta succedendo".
        self.shared_layers = nn.Sequential( # The value a Sequential provides over manually calling a sequence of modules is that it allows treating the whole container as a single module
            nn.Linear(input_dim, 256), # Applies an affine linear transformation to the incoming data: y = x(A)T + b
            nn.ReLU(), # Funzione di attivazione: ReLU(x) = max(0,x) -> spegne i neuroni non importanti (negativi)
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128), #Riassume in 128 valori essenziali
            nn.ReLU()
        )
        
        # 2. L'ATTORE (Policy Head)
        # Sputa fuori 40 numeri (logits). Più alto è il numero, più la rete "vuole" giocare quella carta.
        self.actor = nn.Linear(128, action_dim)
        
        # 3. IL CRITICO (Value Head)
        # Sputa fuori un solo numero: il punteggio finale previsto.
        self.critic = nn.Linear(128, 1)

    def forward(self, x, mask=None):
        """
        Il flusso di informazioni:
        x: l'osservazione da 132 elementi
        mask: la maschera delle azioni legali (action_masks())
        """
        # Passiamo i dati attraverso i layer comuni
        hidden = self.shared_layers(x)
        
        # Calcoliamo le probabilità delle mosse
        policy_logits = self.actor(hidden)
        
        # --- ACTION MASKING ---
        if mask is not None:
            # Se una carta non è in mano (mask=False), mettiamo un valore 
            # bassissimo (-infinito) così la probabilità diventa 0%
            fill_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(~mask, fill_value)
        
        # Trasformiamo i numeri in probabilità (che sommano a 100%)
        probs = F.softmax(policy_logits, dim=-1)
        
        # Il critico valuta lo stato
        value = self.critic(hidden)
        
        return probs, value