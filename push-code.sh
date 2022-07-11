find . -name '*.pyc' -delete
find . -name '*.pyo' -delete
find . -name '.DS_Store' -delete

# update info

git config --global credential.helper store

git config user.email "drssth@gmail.com"
git config user.name "drssth"

git add --all
git commit -m "edit"

git pull
git push
