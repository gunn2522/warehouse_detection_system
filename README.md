# 🏭 Warehouse Detection System

A real-time object detection system for warehouse environments using **YOLOv8**, **Roboflow**, **OpenCV**, and **Django**. It detects and counts objects like **cylinders** and **trucks** from a live RTSP camera feed or video file and displays results on a live web dashboard.

---

## 📸 Demo

![detect](https://github.com/user-attachments/assets/6aa729eb-91b0-4c5c-9b0c-842460a0112a)

---

## 🚀 Features

- 🔍 Detects trucks, cylinders, and more using a custom-trained Roboflow model  
- 🎥 Supports RTSP camera streams and local video files  
- 📦 Real-time object counting and classification  
- 📊 Django-based web dashboard for live statistics  
- 💾 Automatic logging to a database  
- 🌐 Ready for deployment (Heroku, Vercel, PythonAnywhere)  

---

## 🧰 Tech Stack

- **Frontend**: HTML, CSS, Bootstrap/Tailwind  
- **Backend**: Django, Python  
- **Computer Vision**: OpenCV, Roboflow (YOLOv8)  
- **Database**: SQLite (default), PostgreSQL (optional)  
- **Hosting**: Localhost / Vercel / Render / PythonAnywhere  

---

## 📂 Folder Structure

```
warehouse_detection/
├── inventory/
│   ├── templates/
│   │   └── dashboard.html
│   ├── static/
│   │   └── images/
│   │       └── demo.png
│   ├── camera_detection.py
│   ├── views.py
│   └── models.py
├── warehouse_detection/
│   └── settings.py
├── manage.py
├── requirements.txt
└── README.md
```


## 🖼️ Dashboard Features

- Displays **live camera feed** with detections  
- Shows **real-time count** of detected items (cylinders, trucks)  
- Saves **detection data in database** for analytics  

---

## 📈 Future Enhancements

- [ ] Add user authentication  
- [ ] Integrate CSV export of logs  
- [ ] Add email/SMS alerts on threshold  
- [ ] Support multiple camera feeds  

---

## 🧠 Model Training (Optional)

To train a new model using Roboflow:

1. Upload images to [https://roboflow.com](https://roboflow.com)  
2. Annotate and train with YOLOv8  
3. Deploy the model and get your API key  
4. Use the new model ID in `camera_detection.py`  

---

## 🙋‍♂️ Author

**Gunn Malhotra**  
Student | Developer | AI Explorer  
📫 gunnmlhtr@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/gunn-malhotra)  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐️ Star the Repo

If you find this project helpful, consider giving it a ⭐ on GitHub.  
Your support motivates continued development!
