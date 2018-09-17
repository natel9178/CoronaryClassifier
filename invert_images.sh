cd ./data/train

for i in *.jpg; do
    [ -f "$i" ] || break
    convert $i -negate "${i%.*}-negated.jpg"
done