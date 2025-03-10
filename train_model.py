import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# โหลดชุดข้อมูล MNIST (ตัวเลข 0-9)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ปรับขนาดข้อมูลให้อยู่ในช่วง 0-1 และ reshape เป็น 4 มิติ (เพื่อให้ใช้กับ CNN)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ✅ ใช้ Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล
datagen = ImageDataGenerator(
    rotation_range=10,  # หมุนภาพไม่เกิน 10 องศา
    zoom_range=0.1,  # ขยาย-ย่อ 10%
    width_shift_range=0.1,  # เลื่อนแนวนอน
    height_shift_range=0.1  # เลื่อนแนวตั้ง
)
datagen.fit(x_train)  # ปรับข้อมูลการฝึกให้โมเดลเห็นรูปแบบที่หลากหลายขึ้น

# ✅ สร้างโมเดล CNN
model = keras.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # เพิ่ม Dropout ป้องกัน Overfitting
    layers.Dense(10, activation='softmax')
])


# ✅ คอมไพล์โมเดล
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ ฝึกสอนโมเดล
print("🔄 กำลังฝึกโมเดล...")
model.fit(datagen.flow(x_train, y_train, batch_size=32),  # ใช้ข้อมูลที่ถูก Augment
          validation_data=(x_test, y_test),
          epochs=10)

# ✅ บันทึกโมเดลที่ฝึกแล้ว
model.save("mnist_model.h5")
print("✅ บันทึกโมเดลสำเร็จ! (mnist_model.h5)")
