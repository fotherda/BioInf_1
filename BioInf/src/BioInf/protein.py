'''
Created on 10 Mar 2017

@author: Dave
'''
import re
import numpy as np

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData


from cx_Freeze.windist import sequence

WHOLE_SEQUENCE = 0
N_TERMINAL_50 = 1
C_TERMINAL_50 = 2

AA_CODES = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
UNKNOWNS = ['X','U','B']

#In vivo half-life in mammals in hours 
in_vivo_half_life = { 'A': 4.4, 'R': 1, 'N': 1.4, 'D': 1.1, 'C': 1.2,
                      'Q': 0.8, 'E': 1, 'G': 30, 'H': 3.5, 'I': 20,
                      'L': 5.5, 'K': 1.3, 'M': 30, 'F': 1.1, 'P': 20,
                      'S': 1.9, 'T': 7.2, 'W': 2.8, 'Y': 2.8, 'V': 100}

class Protein(ProteinAnalysis):

    category_codes = {'cyto': 0,'mito': 1,'secreted': 2,'nucleus': 3, 'blind': 4}
    inv_category_codes = {0:'cyto',1:'mito',2:'secreted',3:'nucleus', 4:'blind'}
        
    def __init__(self, category_name, name, sequence):
        super(Protein, self).__init__(sequence)
        self._name = name
        self._category = Protein.category_codes[category_name]
        self._sequence = sequence
        if 'X'  in sequence or 'U'  in sequence or 'B' in sequence:
            self._contains_unknown=True
            new_seq = self._sequence.replace('U', '')
            new_seq = new_seq.replace('B', '')
            new_seq = new_seq.replace('X', '')
            self._no_unknowns = ProteinAnalysis(new_seq)
        else:
            self._contains_unknown=False
            self._no_unknowns = self
                
        
    def get_sequence(self):
        return self._sequence    
        
    def sequence_length(self):
        return len(sequence)
    
    def get_sub_sequence(self, seq_flag):
        if seq_flag == WHOLE_SEQUENCE:
            return self
        elif seq_flag == N_TERMINAL_50:
            return Protein(Protein.inv_category_codes[self._category], self._name, self._sequence[:50])
        elif seq_flag == C_TERMINAL_50:
            return Protein(Protein.inv_category_codes[self._category], self._name, self._sequence[-50:])
        
        
    def get_category(self):
        return self._category
    
    def molecular_weight(self): #overrides base class
        if self._contains_unknown:
            new_seq = ''
            for aa in self._sequence:
                if aa not in UNKNOWNS:
                    new_seq +=  aa
            new_p = ProteinAnalysis(new_seq)
            mw = new_p.molecular_weight()
            #just increase by avg mw of known aa's
            mw *= len(self._sequence)/len(new_seq)
        else:
            mw = super(Protein, self).molecular_weight()
         
        return mw    
                       
    def instability_index(self): #overrides base class
        if self._contains_unknown:
            index = ProtParamData.DIWV 
            score = 0.0 
            sub_sequences = self._sequence.replace('U', 'X')
            sub_sequences = sub_sequences.replace('B', 'X')
            sub_sequences = sub_sequences.split('X')
            for seq in sub_sequences:
                for i in range(len(seq) - 1): 
                    this, nextt = seq[i:i + 2] 
                    dipeptide_value = index[this][nextt] 
                    score += dipeptide_value 

            in_idx = (10.0 / (len(self._sequence)-len(sub_sequences) + 1)) * score                                
        else:
            in_idx = super(Protein, self).instability_index()
         
        return in_idx    
                       
    def flexibility(self): #overrides base class
        if self._contains_unknown:
            new_sequence = self._sequence.replace('X', 'R') #replace with an avg aa
            new_sequence = new_sequence.replace('U', 'R') #replace with an avg aa
            new_sequence = new_sequence.replace('B', 'R') #replace with an avg aa
            new_p = ProteinAnalysis(new_sequence)
            flex = new_p.flexibility()
        else:
            flex = super(Protein, self).flexibility()
         
        return flex
                       
    def in_vivo_half_life(self): #N-end rule
        if self._sequence[0] not in UNKNOWNS:
            return in_vivo_half_life[ self._sequence[0] ]
        else:
            return 5 #approx avg value
        
    
    def has_KDEL(self): #ER retention signal
        if self._sequence[-4:] == 'KDEL':
            return True
        else:
            return False
        
    def has_KKXX(self): #ER retention signal
        if self._sequence[-4:-2] == 'KK':
            return True
        else:
            return False
  
    def has_NLS(self): #NLS signal
        if 'PKKKRKV' in self._sequence:
            return True
        else:
            return False
    
    def has_Chelsky_sequence(self): #NLS signal
#               K-K/R-X-K/R
        try:
            found = re.search('K[KR].[KR]', self._sequence)
        except AttributeError:
            # AAA, ZZZ not found in the original string
            found = None # apply your error handling
        
        if found is not None:
            return True
        else:
            return False           
    
    def has_PTS(self): #peroxisomal targeting signal (PTS) - SKL in carboxy tail
        if self._sequence[-3:] == 'SKL':
            return True
        else:
            return False
    
    def hydrophobicity(self):
        hphob = self._no_unknowns.protein_scale(ProtParamData.kd, 3)
        return np.mean(hphob)
    
    def surface_accessibility(self):
        em = self._no_unknowns.protein_scale(ProtParamData.em, 3)
        return np.mean(em)
    
    def transfer_energy(self):
        te = self._no_unknowns.protein_scale(ProtParamData.ja, 3)
        return np.mean(te)
    
    def hydrophilicity(self):
        hphil = self._no_unknowns.protein_scale(ProtParamData.hw, 3)
        return np.mean(hphil)
    
            

        