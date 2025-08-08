import cv2
import mediapipe as mp
import numpy as np
import json
import os

# === 전체 처리 함수 ===
def run_human_to_giraffe(HUMAN_IMAGE_PATH):
    # === 파일 경로 설정 ===
    GIRAFFE_IMAGE_PATH = 'giraffe.jpg'
    GIRAFFE_EYE_JSON_PATH = 'giraffe_eye_points.json'

    # === mediapipe 초기화 ===
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    # === 눈 / 눈썹 인덱스 설정 ===
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144, 163, 7]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373, 390, 249]
    PRESERVE_LEFT = [133, 159]
    PRESERVE_RIGHT = [362, 386]
    LEFT_BROW_IDX = [70, 63, 105, 66, 107]
    RIGHT_BROW_IDX = [336, 296, 334, 293, 300]

    # === 눈 추출 함수 (경계 부드럽게) ===
    def get_expanded_eye(img, landmarks, eye_idx, preserve_idx, expand_ratio=1.1):
        h, w = img.shape[:2]
        center_x, center_y = 0, 0
        points = []

        for idx in eye_idx:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            center_x += x
            center_y += y

        center_x //= len(points)
        center_y //= len(points)

        expanded_points = []
        for (x, y), idx in zip(points, eye_idx):
            if idx in preserve_idx:
                expanded_points.append((x, y))
            else:
                dx = x - center_x
                dy = y - center_y
                new_x = int(center_x + dx * expand_ratio)
                new_y = int(center_y + dy * expand_ratio)
                expanded_points.append((new_x, new_y))

        # 마스크 생성
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(expanded_points, dtype=np.int32)], 255)

        # 부드러운 경계 적용
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # 눈 영역 추출
        eye_only = cv2.bitwise_and(img, img, mask=blurred_mask)
        eye_bgra = cv2.cvtColor(eye_only, cv2.COLOR_BGR2BGRA)
        eye_bgra[:, :, 3] = blurred_mask  # 부드러운 마스크를 알파 채널로

        # 눈 크롭
        x, y, w_box, h_box = cv2.boundingRect(np.array(expanded_points))
        eye_crop = eye_bgra[y:y + h_box, x:x + w_box]
        eye_center = (center_x - x, center_y - y)

        # 💡 눈 확대
        scale = 1.4
        eye_crop = cv2.resize(eye_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        eye_center = (int(eye_center[0] * scale), int(eye_center[1] * scale))

        return eye_crop, eye_center

    # === 눈썹 추출 함수 ===
    def get_eyebrow_patch(img, landmarks, brow_idx, expand=1.2):
        h, w = img.shape[:2]
        points = []
        center_x, center_y = 0, 0

        for idx in brow_idx:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            center_x += x
            center_y += y

        center_x //= len(points)
        center_y //= len(points)

        # 확장
        expanded = []
        for (x, y) in points:
            dx = x - center_x
            dy = y - center_y
            new_x = int(center_x + dx * expand)
            new_y = int(center_y + dy * expand)
            expanded.append((new_x, new_y))

        # 마스크 만들기
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(expanded, dtype=np.int32)], 255)
        blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)

        brow_rgb = cv2.bitwise_and(img, img, mask=blurred_mask)
        brow_bgra = cv2.cvtColor(brow_rgb, cv2.COLOR_BGR2BGRA)
        brow_bgra[:, :, 3] = blurred_mask

        # 크롭
        x, y, w_box, h_box = cv2.boundingRect(np.array(expanded))
        brow_crop = brow_bgra[y:y + h_box, x:x + w_box]
        brow_center = (center_x - x, center_y - y)

        return brow_crop, brow_center

    # === 눈 붙이기 함수 ===
    def paste_eye(eye_img, eye_center, target_pos, canvas):
        h, w = eye_img.shape[:2]
        cx, cy = eye_center
        tx, ty = target_pos
        x1 = tx - cx
        y1 = ty - cy
        x2 = x1 + w
        y2 = y1 + h

        if x1 < 0 or y1 < 0 or x2 > canvas.shape[1] or y2 > canvas.shape[0]:
            print("⚠️ 경계 벗어남:", (x1, y1), (x2, y2))
            return

        roi = canvas[y1:y2, x1:x2]
        mask = eye_img[:, :, 3].astype(float) / 255.0  # 알파 채널 정규화
        mask = cv2.merge([mask] * 3)
        eye_rgb = eye_img[:, :, :3].astype(float)

        # 블렌딩
        blended = eye_rgb * mask + roi.astype(float) * (1 - mask)
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    # === 눈 위로 살짝 이동해서 붙이기 ===
    def paste_above(eye_pos, offset=30):  # offset은 기린 해상도에 따라 조절
        return (eye_pos[0], eye_pos[1] - offset)

    # === 사람 얼굴 분석 ===
    img_human = cv2.imread(HUMAN_IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_human, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        raise RuntimeError("사람 얼굴 인식 실패")
    face_landmarks = results.multi_face_landmarks[0]

    # 눈 2개 추출
    left_eye_img, left_center = get_expanded_eye(img_human, face_landmarks.landmark, LEFT_EYE_IDX, PRESERVE_LEFT)
    right_eye_img, right_center = get_expanded_eye(img_human, face_landmarks.landmark, RIGHT_EYE_IDX, PRESERVE_RIGHT)

    # === 눈썹 추출 ===
    left_brow_img, left_brow_center = get_eyebrow_patch(img_human, face_landmarks.landmark, LEFT_BROW_IDX)
    right_brow_img, right_brow_center = get_eyebrow_patch(img_human, face_landmarks.landmark, RIGHT_BROW_IDX)

    # === 기린 이미지 및 붙일 좌표 로드 ===
    img_giraffe = cv2.imread(GIRAFFE_IMAGE_PATH)
    with open(GIRAFFE_EYE_JSON_PATH, "r") as f:
        giraffe_eye_points = json.load(f)

    if len(giraffe_eye_points) != 2:
        raise ValueError("기린 눈 중심 좌표 2개가 필요합니다.")

    giraffe_left = giraffe_eye_points[1]
    giraffe_right = giraffe_eye_points[0]

    # === 실제 붙이기 ===
    paste_eye(left_eye_img, left_center, giraffe_left, img_giraffe)
    paste_eye(right_eye_img, right_center, giraffe_right, img_giraffe)

    # === 눈 위로 살짝 이동해서 눈썹 붙이기 ===
    paste_eye(left_brow_img, left_brow_center, paste_above(giraffe_left), img_giraffe)
    paste_eye(right_brow_img, right_brow_center, paste_above(giraffe_right), img_giraffe)

    # === 결과 저장 ===
    output_path = os.path.join('uploads', 'result.jpg')
    cv2.imwrite(output_path, img_giraffe)
    return output_path
