import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

# โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model("mnist_model.h5")

# ฟังก์ชันให้ทำนายตัวเลขจากภาพ
def predict_digit(img):
    img = img.resize((28, 28)).convert('L')  # Resize และแปลงเป็นขาวดำ
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape สำหรับโมเดล
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)  # คืนค่าหมายเลขที่โมเดลทำนาย และค่าความมั่นใจ

# ฟังก์ชันเมื่อกดปุ่ม "ทายเลข"
def classify():
    digit, confidence = predict_digit(image)
    label_result.config(text=f"🔢 ทำนายว่าเป็นเลข: {digit}\n📊 ความน่าจะเป็น: {confidence:.2%}", foreground="green")

# ฟังก์ชันเคลียร์ Canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="black")
    label_result.config(text="✏️ วาดตัวเลข แล้วกดปุ่ม 'ทายเลข'", foreground="black")

# ฟังก์ชันวาดเส้น
def draw_lines(event):
    x, y = event.x, event.y
    r = 10  # ขนาดเส้นใหญ่ขึ้น
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill="white")

# =================== UI Design ===================
root = tk.Tk()
root.title("🖌️ AI คาดเดาตัวเลข")
root.geometry("400x500")  # ปรับขนาดหน้าต่าง
root.configure(bg="#F4F4F4")  # เปลี่ยนสีพื้นหลังให้ดูสว่างขึ้น

# ใช้ ttk.Style เพื่อปรับแต่ง UI
style = ttk.Style()
style.configure("TButton", font=("Arial", 14), padding=10)
style.configure("TLabel", font=("Arial", 14))

# สร้างเฟรมหลัก
frame = ttk.Frame(root, padding=10)
frame.pack(pady=10)

# Canvas สำหรับวาดตัวเลข
canvas = tk.Canvas(frame, width=280, height=280, bg="black", bd=3, relief="ridge")
canvas.grid(row=0, column=0, columnspan=2, pady=10)
canvas.bind("<B1-Motion>", draw_lines)

# สร้าง Image สำหรับวาด
image = Image.new("RGB", (280, 280), "black")
draw = ImageDraw.Draw(image)

# Label แสดงผล
label_result = ttk.Label(frame, text="✏️ วาดตัวเลข แล้วกด 'ทายเลข'", foreground="black")
label_result.grid(row=1, column=0, columnspan=2, pady=10)

# ปุ่ม "ทายเลข"
btn_classify = ttk.Button(frame, text="🔍 ทายเลข", command=classify)
btn_classify.grid(row=2, column=0, padx=5, pady=5, ipadx=10, ipady=5)

# ปุ่ม "ล้าง"
btn_clear = ttk.Button(frame, text="🧹 ล้าง", command=clear_canvas)
btn_clear.grid(row=2, column=1, padx=5, pady=5, ipadx=10, ipady=5)

# เริ่มโปรแกรม
root.mainloop()
