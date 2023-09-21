import pandas as pd
from googletrans import Translator

class tradutorCSV:
   def __init__(self, arquivo_entrada, colunas_a_traduzir, arquivo_saida):
        self.arquivo_entrada = arquivo_entrada
        self.colunas_a_traduzir = colunas_a_traduzir
        self.arquivo_saida = arquivo_saida
        self.translator = Translator()

   

        
