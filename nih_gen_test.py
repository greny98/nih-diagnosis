from nih.data_generator import read_csv, train_val_split, ClassifyGenerator
from nih.model import create_nih_model

X_train_val_df, y_train_val_df = read_csv('data/nih/train_val.csv')
# Split train, val
(X_train, y_train), (X_val, y_val) = train_val_split(X_train_val_df.values, y_train_val_df.values, log=False)
X_train = X_train.reshape(-1)
X_val = X_val.reshape(-1)
train_ds = ClassifyGenerator(X_train, y_train, 'data/nih/images/', training=True, batch_size=4)

for x in train_ds:
    print(x)

# model = create_nih_model()
# model.summary()
