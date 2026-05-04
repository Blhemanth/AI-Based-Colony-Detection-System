from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\heman\Desktop\bacteria_detection\runs\detect\train-3\weights\best.pt")

image_path = r"C:\Users\heman\Desktop\bacteria_detection\dataset\images\val\480_jpg.rf.9177d7999b4062fe4110b0d94d571203.jpg"

results = model(image_path, conf=0.5)

for r in results:
    count = 0

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)

        if float(box.conf[0]) > 0.5 and area > 50:
            count += 1

    print("🦠 Total Colonies:", count)

    img = r.plot()

    cv2.putText(img, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()