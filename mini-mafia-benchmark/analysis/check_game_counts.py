import sqlite3
import pandas as pd
from collections import Counter

db_path = '../database/mini_mafia.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql_query('''
    SELECT g.game_id, g.winner, gp.role, p.model_name
    FROM games g
    JOIN game_players gp ON g.game_id = gp.game_id
    JOIN players p ON gp.player_id = p.player_id
''', conn)
conn.close()

mapping = {
    'claude-opus-4-1-20250805': 'Claude Opus 4.1',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'deepseek-chat': 'DeepSeek V3.1',
    'deepseek-reasoner': 'DeepSeek V3.1',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
    'gpt-4.1-mini': 'GPT-4.1 Mini',
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt-5': 'GPT-5 Mini',
    'grok-3-mini': 'Grok 3 Mini',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B Instruct',
    'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
    'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
}
df['display_name'] = df['model_name'].map(mapping)

BACKGROUNDS = ['DeepSeek V3.1', 'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'Mistral 7B Instruct']

maf_counts = Counter()
det_counts = Counter()
vil_counts = Counter()

for game_id in df['game_id'].unique():
    game_df = df[df['game_id'] == game_id]
    roles = {}
    for _, row in game_df.iterrows():
        if pd.notna(row['display_name']):
            roles[row['role']] = row['display_name']

    if len(roles) == 3:
        maf = roles.get('mafioso')
        det = roles.get('detective')
        vil = roles.get('villager')
        if maf and det and vil:
            if (det == vil and det in BACKGROUNDS) or (maf == det and maf in BACKGROUNDS) or (maf == vil and maf in BACKGROUNDS):
                maf_counts[maf] += 1
                det_counts[det] += 1
                vil_counts[vil] += 1

print('MAFIOSO:')
for m in sorted(maf_counts.keys()):
    print(f'  {m}: {maf_counts[m]}')

print('\nDETECTIVE:')
for m in sorted(det_counts.keys()):
    print(f'  {m}: {det_counts[m]}')

print('\nVILLAGER:')
for m in sorted(vil_counts.keys()):
    print(f'  {m}: {vil_counts[m]}')
