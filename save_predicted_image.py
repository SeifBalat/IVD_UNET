# Perform a sanity check on some random validation samples

x = 0
for x in range(1, len(preds_test_t)-1):
  ix_val = x
  print(ix_val)
  imshow(X_train[int(X_train.shape[0]*0.9):][ix_val])
  plt.show()
  plt.imsave('a1.png' , X_train[int(X_train.shape[0]*0.9):][ix_val])
  imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix_val]))
  plt.show()
  plt.imsave('a2.png' , X_train[int(X_train.shape[0]*0.9):][ix_val])
  imshow(np.squeeze(preds_val_t[ix_val]))
  plt.show()
  plt.imsave('a3.png' , np.squeeze(preds_val_t[ix_val]) )
    #im1 = Image.open('a1.jpg')
  #im2 = Image.open('a2.jpg')
  #im3 = Image.open('pil3.jpg')
  list_im = ['a1.png', 'a2.png', 'a3.png']
  imgs    = [ Image.open(i) for i in list_im ]
  # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
  min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
  imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
  imgs_comb = Image.fromarray( imgs_comb)
  imgs_comb = imgs_comb.convert('RGB')
  imgs_comb.save( 'Trifecta.jpg' )    

# for a vertical stacking it is simple: use vstack
  imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
  imgs_comb = Image.fromarray( imgs_comb)
  imgs_comb = imgs_comb.convert('RGB')
  imgs_comb.save( '/content/drive/My Drive/op_test/Trifecta_vertical'+ str(x) + '.jpg' )
  x+=1