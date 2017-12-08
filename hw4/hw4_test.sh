wget -O model.h5 'https://www.dropbox.com/s/xrgvkiai75smm7d/model.h5?dl=1'
python3 preprocessForTest.py $1
python3 main_testing.py $2