import json
from tkinter import Frame, Tk, BOTH


import tkinter as tk
from tkinter import filedialog
from segmentation import get_image_from_bytes, get_yolov5

model = get_yolov5()
class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("ImageShape")
        self.pack(fill=BOTH, expand=1)

        Button = tk.Button(self, text="Tìm biên", command=self.detect_food_return_json_result, bg='blue', fg='white')
        Button.place(x=50, y=60)

    def detect_food_return_json_result(self):
        data = 'cars'
        chose_img = filedialog.askopenfilename()
        input_image = get_image_from_bytes(chose_img)
        results = model(input_image)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
        for i in range(len(detect_res)):
            if detect_res[i].get("name") == data:
                detect_res = json.loads(detect_res)

        return {"result": detect_res}

root = Tk()
root.geometry("500x500+300+300")
ex = Example(root)
root.mainloop()