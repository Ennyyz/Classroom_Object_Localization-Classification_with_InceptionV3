from model.ciou_loss import ciou_loss
from model.InceptionV3 import InceptionV3

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy,CategoricalCrossentropy

model = InceptionV3()

lr = 1e-4  
num_epochs = #your_expect_epochs

model.compile(
    loss = {
#         "bbox": BinaryCrossentropy(from_logits=False),
        "bbox": ciou_loss,
        "label": CategoricalCrossentropy(from_logits=False) ,
    },
    optimizer=Adam(lr),
    metrics={
        "bbox": ['acc'], 
        "label": ['acc'] 
    }
)

callbacks = [
    ModelCheckpoint('best_model.keras', verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-6, verbose=1),
    CSVLogger('log.csv', append=True),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False, verbose=1),
]

history= model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=valid_ds,
    callbacks=callbacks
)
model.save('final_model.keras')
