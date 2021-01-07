Fichier `Fish.csv`

Chargement des données:
```R
Fish <- read.table("Fish.csv", sep = ";", header=TRUE)
```

On a mesuré des caractéristiques physiques (`Weight`, `Height` et `Width`) de différents poissons. On cherche à étudier les poissons d'une certaine espèce (variable `Species`: `1` pour des poissons de l'espèce en question, `0` pour des poissons d'autres espèces).

Question : Peut-on prédire si les poissons appartiennent à l'espèce étudiée en fonction de leur mensuration ?
