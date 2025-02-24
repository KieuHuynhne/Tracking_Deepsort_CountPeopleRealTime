from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import config


def _display_detected_frames(conf, model, st_count, st_frame, image):
    """
    Hiển thị các đối tượng được phát hiện trên video bằng YOLOv8.
    :param conf (float): Confidence threshold.
    :param model: YOLOv8
    :param st_frame (Streamlit object): Đối tượng Streamlit để hiển thị video được phát hiện.
    :param image (numpy array): Mảng numpy để hiển thị video
    :return: None
    """

    #Dự đoán đối tượng
    res = model.predict(image, conf=conf)

    # Hiển thị vạch kẻ
    cv2.line(image,(800, 100), (800, 600), (46, 162, 112), 3)

    inText = 'Lên xe'
    outText = 'Xuống xe'
    if config.OBJECT_COUNTER1 != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER1.items()):
            inText += ' - ' + str(key) + ": " +str(value)
    if config.OBJECT_COUNTER != None:
        for _, (key, value) in enumerate(config.OBJECT_COUNTER.items()):
            outText += ' - ' + str(key) + ": " +str(value)
    
    # Plot the detected objects on the video frame
    st_count.write(inText + '\n\n' + outText)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource  #Decorator giúp lưu trữ mô hình để không phải tải lại mỗi khi ứng dụng chạy
def load_model(model_path): #Tải model
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):  #Hàm xử lý ảnh tải lên

    source_img = st.sidebar.file_uploader(
        label="Chọn 1 hình ảnh...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2) #Tạo 2 cột, col1 hiển thị ảnh gốc, col2 hiển thị ảnh đã detect

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img) 
            # Thêm hình ảnh đã tải lên có chú thích
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1] #Vẽ bb lên ảnh để đánh dấu đối tượng đã detect
                                                        #[:, :, ::-1] đảo ngược kênh màu từ BGR sang RGB vì OpenCV sử dụng định dạng BGR trong khi PIL sử dụng RGB
                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):  #Tạo expandable block hiển thị thông tin chi tiết về các đối tượng đã phát hiện
                            for box in boxes:
                                st.write(box.xywh)  # Hiển thị tọa độ và kích thước
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)


def infer_uploaded_video(conf, model):      #Hàm xử lý video tải lên
    num_students_on_board = st.number_input("Số lượng học sinh đi xe hôm nay: ", min_value=1)

    source_video = st.sidebar.file_uploader(
        label="Chọn 1 video..."
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    config.OBJECT_COUNTER1 = {}   #Đặt lại bộ đếm đối tượng để bắt đầu tính toán mới
                    config.OBJECT_COUNTER = {}
                    tfile = tempfile.NamedTemporaryFile() #Tạo 1 tệp tạm để tải video lên
                    tfile.write(source_video.read()) #Ghi nội dung video tải lên vào tệp tạm 
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_count = st.empty() # Tạo chỗ trống giao diện
                    st_frame = st.empty()

                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read() #đọc từng khung hình video, image dưới dạng numpy
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_count,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                    # Đếm số người xuống xe sau khi video chạy xong
                    total_people_out = sum(config.OBJECT_COUNTER.values())

                    # So sánh số lượng nhập ban đầu và số người xuống xe
                    if total_people_out != num_students_on_board:
                        st.warning(f"Số người xuống xe ({total_people_out}) không khớp với số người ban đầu ({num_students_on_board}). Bạn cần kiểm tra lại!")
                    else:
                        st.success(f"Tất cả học sinh đã xuống xe. Tổng số: {total_people_out}")
                    
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model): #Hàm xử lý webcam
    num_students_on_board = st.number_input("Số lượng học sinh đi xe hôm nay: ", min_value=1)
    try:
        config.OBJECT_COUNTER1 = {}   #Đặt lại bộ đếm đối tượng để bắt đầu tính toán mới
        config.OBJECT_COUNTER = {}
        # flag = st.button(
        #     label="Tạm dừng"
        # )
        # Sử dụng checkbox để dừng webcam
        stop_button = st.checkbox(label="Tạm dừng", value=False)
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_count = st.empty()
        st_frame = st.empty()
        while not stop_button:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_count,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
        vid_cap.release()
        # Đếm số người xuống xe sau khi video chạy xong
        total_people_out = sum(config.OBJECT_COUNTER.values())

        # So sánh số lượng nhập ban đầu và số người xuống xe
        if total_people_out != num_students_on_board:
            st.warning(f"Số người xuống xe ({total_people_out}) không khớp với số người ban đầu ({num_students_on_board}). Bạn cần kiểm tra lại!")
        else:
            st.success(f"Tất cả học sinh đã xuống xe. Tổng số: {total_people_out}")

    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
