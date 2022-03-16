import matplotlib.pyplot as plt
import pandas as pd

"""
df = pd.read_csv('wmt2014.tsv',sep='\t')


plt.rcParams["figure.figsize"] = (30, 14)
plt.bar(df.index, df['BLEU'])
plt.xticks(df.index, df['name'], size=16)
plt.yticks(size=20)
plt.ylim(33, 47)
plt.ylabel('Score BLEU', size=20)
plt.savefig('bleu.pdf')
plt.close()
"""


df = pd.read_csv("wmt2014-half.tsv", sep="\t")


fig, ax = plt.subplots(figsize=(17, 9))
ax.set_title(
    "Traduction automatique sur le jeu de données WMT2014 English-French", size=25
)
ax.bar(df.index, df["BLEU"])
ax.set_xticks(df.index)
ax.set_xticklabels(df["name"].values)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=20)
ax.set_ylim(35, 47)
ax.set_ylabel("Score BLEU", size=20)

for i, v in enumerate(df["BLEU"]):
    ax.text(i - 0.25, v - 0.7, str(v), color="white", size=20)

fig.savefig("bleu.pdf")
