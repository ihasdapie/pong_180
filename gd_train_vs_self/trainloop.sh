# there is a memory issue with the training loop
# that tf.reset_default_graphs
# or keras.backend.reset_session
# doesn't seem to fix. Hence this hacky-ass solution



while true; do
    python3 Pong_Game.py
    cp -r /content/pong_180/gdRlbot/mdls/ /content/drive/MyDrive/pong/gd/
    cp -r /content/pong_180/gdRlbot/data/ /content/drive/MyDrive/pong/gddata/ 
     
done
