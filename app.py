from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import os
from eyeMask import run_human_to_giraffe  # ✅ 기린 합성 함수 (human.jpg → result.jpg 저장)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === 1. 업로드 페이지 ===
@app.route('/')
def show_upload():
    return render_template('upload.html')  # templates/upload.html 사용

# === 2. 업로드 처리 및 기린 합성 ===
@app.route('/upload', methods=['POST'])
def upload_image():
    # ✅ 파일 유무 확인
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '이미지 파일이 포함되지 않았습니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다.'}), 400

    # ✅ 파일 저장
    save_path = os.path.join(UPLOAD_FOLDER, 'human.jpg')
    file.save(save_path)

    # ✅ 합성 처리
    try:
        run_human_to_giraffe(save_path)  # 결과는 uploads/result.jpg로 저장된다고 가정
    except Exception as e:
        return jsonify({'success': False, 'error': f'기린 합성 실패: {str(e)}'}), 500

    # ✅ 성공 응답 반환 → JS에서 editorUrl로 redirect
    return jsonify({
        'success': True,
        'originalName': file.filename,
        'editorUrl': url_for('show_result')  # /index 페이지 URL
    })

# === 3. 결과 페이지 (기린 합성 결과 보여줌) ===
@app.route('/index')
def show_result():
    return render_template('index.html')  # 하드코딩된 /uploads/result.jpg 사용

# === 4. 업로드된 파일 제공 (이미지 표시용) ===
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# === 5. 실행 ===
if __name__ == '__main__':
    app.run(debug=True)
