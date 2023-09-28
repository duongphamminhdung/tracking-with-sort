echo config git
git config --global user.email "duongphamminhdung@gmail.com"
git config --global user.name "duongphamminhdung"

echo pip install -q condacolab
pip install -q condacolab

echo installing nvtop
apt install nvtop

echo installing anaconda
python iconda.py

echo init conda
conda init 

echo conda create --name simp python=3.7
conda create --name simp python=3.7

tmux
