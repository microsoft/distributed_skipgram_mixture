# build word_embedding

git clone https://github.com/Microsoft/multiverso

cd multiverso
cd third_party
sh install.sh
cd ..
make -j4 all

cd ..
make -j4
