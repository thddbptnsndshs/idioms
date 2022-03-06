import pandas as pd
import numpy as np
import csv

class PcibexFarm:
    
    def __init__(self, filename, tag='F', ungram='u', gram='g'):
        
        self.tag = tag
        self.gram = gram
        self.ungram = ungram
        
        content = []
        fieldnames = ['reception_time', 'ip', 'controller',
                     'order', 'n', 'label', 'lsg', 'sentence', 'answer',
                     'correct', 'time']

        with open(filename, 'r') as f:
            rdr = csv.DictReader(filter(lambda row: row[0]!='#', f), fieldnames=fieldnames)
            for line in rdr:
                content.append(line)
        
        df = pd.DataFrame.from_dict(content)
        self.df = df
            
    def fit(self, stdise=False, wrong_ips=False, stdise_columns=['answer', 'time']):

        data = self.df.groupby('sentence').bfill()
        data.drop(data[data['correct'] != 'NULL'].index, inplace=True)
        data.drop(data[data['controller'] != 'AcceptabilityJudgment'].index, inplace=True)
        data.drop(columns=['correct', 'order', 'n', 'lsg', 'reception_time', 'controller'], inplace=True)
        data['answer'] = data['answer'].astype(int)
        data['time'] = data['time'].astype(int)
        
        if wrong_ips:
            
            fillers = data[data['label'].str.startswith(self.tag)]
            fillers['grammatical'] = fillers['label'].str[1]
            fillers['grammatical'] = fillers['grammatical'].map({self.ungram:0, self.gram:1})
            fillers.drop(columns=['label'], inplace=True)

            ips = pd.pivot_table(fillers, values='answer', index='ip', columns='grammatical', aggfunc=np.average)
            ips = ips.reset_index().rename_axis(None, axis=1)
            bad = []

            for row in ips.loc[(ips[1] <= 4) | (ips[0] >= 4)].iterrows():
                bad.append(row[0])

            bad = list(bad)
            wrong_ips = list(ips.loc[bad]['ip'])

            for wrong in wrong_ips:
                data.drop(data.loc[data['ip'] == wrong].index, inplace=True)

            self.data = data
                
        if stdise:
            for col in stdise_columns:
                data[col] = data.groupby('ip')[col].transform(lambda x: (x - x.mean()) / x.std())
                 
        self.data = data
        
        return data
    
    def fillers(self):
        
        data = self.data
        fillers = data[data['label'].str.startswith(self.tag)]
        fillers['grammatical'] = fillers['label'].str[1]
        fillers['grammatical'] = fillers['grammatical'].map({self.ungram:0, self.gram:1})
        fillers.drop(columns=['label'], inplace=True)
        
        self.fillers = fillers
        
        return fillers
    
    def test_items(self, tags):
        
        data = self.data
        test_items = data[data['label'].str.startswith(tuple(tags))]
        self.test_items = test_items
        
        return test_items