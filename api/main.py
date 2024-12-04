import base64
from flask import Flask, render_template, request, jsonify
import face_recognition
import os



app = Flask(__name__)

def get_image_names(folder_path):
    return [file_name for file_name in os.listdir(folder_path) if file_name.endswith(('.jpg', '.jpeg', '.png'))]

def compute_similarity(image_encoding1, image_encoding2):
    return face_recognition.face_distance([image_encoding1], image_encoding2)[0]


def find_best_match(got_image_encodings, folder_path, threshold=0.6):
    best_match = None
    best_similarity = float('inf')  # Initialize with a high value
    for image_name in get_image_names(folder_path):
        existing_image_path = os.path.join(folder_path, image_name)
        existing_image = face_recognition.load_image_file(existing_image_path)
        existing_image_encodings = face_recognition.face_encodings(existing_image)
        if existing_image_encodings:
            for existing_image_encoding in existing_image_encodings:
                for got_image_encoding in got_image_encodings:
                    similarity = compute_similarity(existing_image_encoding, got_image_encoding)
                    if similarity < threshold and similarity < best_similarity:
                        best_similarity = similarity
                        best_match = image_name
    if best_match:
        return best_match
    else:
        return "No matching face found"

def compare_faces(image_data, folder_path):
    try:
        # Decode base64 image data
        got_image = face_recognition.load_image_file(image_data)
        got_image_encodings = face_recognition.face_encodings(got_image)
        if not got_image_encodings:
            return "No face detected in the uploaded image"

        # Call the find_best_match function to find the image with the highest similarity
        result = find_best_match(got_image_encodings, folder_path)
        return result
    except Exception as e:
        return f"An error occurred: {e}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data.get('image_data')
    folder_path = "./images"  # Change this to the path of your image folder

    # Decode base64 image data
    img_data = base64.b64decode(image_data.split(',')[1])

    # Specify the path to save the image
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the image to a file
    image_path = os.path.join(upload_folder, 'captured_image.jpg')
    with open(image_path, 'wb') as f:
        f.write(img_data)

    # Call the compare_faces function to process the image data
    result = compare_faces(image_path, folder_path)

    return jsonify({'message': result})


@app.route('/result')
def result():
    result = request.args.get('result')
    result = result[:len(result)-4]
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
