import cv2
from optical_flow import *
from PIL import Image

def save_c3d_frame_opticalflow_result(input_np, output_np, save_name, save_path='../RESULT/C3_Their/'):
    input_gray_np = input_np[0, :, :, :, 0]
    input_optical_np = input_np[0, :, :, :, 1:]
    output_gray_np = output_np[0, :, :, :, 0]
    output_optical_np = output_np[0, :, :, :, 1:]

    print (output_np.shape)
    for i in range(int(output_gray_np.shape[0])):
        gray_concat_np = np.concatenate([output_gray_np[i, :, :], input_gray_np[i, :, :]], axis=0)

        input_rgb_of_optical_flow = get_rgb_np_of_optical_flow(input_optical_np[i, :])
        output_rgb_of_optical_flow = get_rgb_np_of_optical_flow(output_optical_np[i, :])
        optical_concat_np = np.concatenate([input_rgb_of_optical_flow, output_rgb_of_optical_flow], axis=0)


        img_optical_concat = Image.fromarray(optical_concat_np, mode='RGB')
        img_optical_concat.save(save_path + 'OPTICAL_FLOW/' + save_name + 'frame%d_opticalflow.bmp' % (i + 1))

        cur_gray_output = 255.0 * np.array(gray_concat_np, dtype=float)
        cur_gray_output = cur_gray_output.astype('uint8')
        img_gray = Image.fromarray(cur_gray_output, mode='L')
        img_gray.save(save_path + 'GRAY/' + save_name + 'frame%d_gray.bmp' % (i + 1))

    return

def save_c3d_text_frame_opticalflow_result( input_np, output_np, text,save_name, save_path='../RESULT/C3_Their/'):
    # print input_np.shape
    # print output_np.shape
    input_gray_np = input_np[0, :, :, :, 0]
    input_optical_np = input_np[0, :, :, :, 1:]
    output_gray_np = output_np[0, :, :, :, 0]
    output_optical_np = output_np[0, :, :, :, 1:]

    print (output_np.shape)
    for i in range(int(output_gray_np.shape[0])):
        gray_concat_np = np.concatenate([output_gray_np[i, :, :], input_gray_np[i, :, :]], axis=0)

        input_rgb_of_optical_flow = get_rgb_np_of_optical_flow(input_optical_np[i, :])
        output_rgb_of_optical_flow = get_rgb_np_of_optical_flow(output_optical_np[i, :])
        optical_concat_np = np.concatenate([input_rgb_of_optical_flow, output_rgb_of_optical_flow], axis=0)
        cv2.putText(optical_concat_np, str(text), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(save_path + 'OPTICAL_FLOW/' + save_name + 'frame%d_opticalflow.bmp' % (i + 1),optical_concat_np)

        # img_optical_concat = Image.fromarray(optical_concat_np, mode='RGB')
        # img_optical_concat.save(save_path + 'OPTICAL_FLOW/' + save_name + 'frame%d_opticalflow.bmp' % (i + 1))

        cur_gray_output = 255.0 * np.array(gray_concat_np, dtype=float)

        cv2.putText(cur_gray_output, str(text), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(save_path + 'GRAY/' + save_name + 'frame%d_opticalflow.bmp' % (i + 1),cur_gray_output)

        # cur_gray_output = cur_gray_output.astype('uint8')
        # img_gray = Image.fromarray(cur_gray_output, mode='L')
        # img_gray.save(save_path + 'GRAY/' + save_name + 'frame%d_gray.bmp' % (i + 1))

    return

def save_c3d_text_opticalflow_result( input_np, output_np, text,save_name, save_path='../RESULT/C3_Their/'):
    # print input_np.shape
    # print output_np.shape
    input_optical_np = input_np[0,:]
    output_optical_np = output_np[0,:]

    # for i in range(int(input_optical_np.shape[0])):
    for i in range(int(1)):
        input_rgb_of_optical_flow = get_rgb_np_of_optical_flow(input_optical_np[i, :])
        output_rgb_of_optical_flow = get_rgb_np_of_optical_flow(output_optical_np[i, :])
        optical_concat_np = np.concatenate([input_rgb_of_optical_flow, output_rgb_of_optical_flow], axis=0)
        cv2.putText(optical_concat_np, str(text), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(save_path + 'OPTICAL_FLOW/' + save_name + 'frame%d_opticalflow.bmp' % (i + 1),optical_concat_np)

    return

def save_c3d_text_frame_result( input_np, output_np, text,save_name, save_path='../RESULT/C3_Their/'):
   
    input_gray_np = input_np[0,:]
    output_gray_np = output_np[0,:]

    # print output_np.shape
    # for i in range(int(output_gray_np.shape[0])):
    for i in range(int(1)):
        gray_concat_np = np.concatenate([output_gray_np[i, :, :], input_gray_np[i, :, :]], axis=0)
        cur_gray_output = 255.0 * np.array(gray_concat_np, dtype=float)
        cv2.putText(cur_gray_output, str(text), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imwrite(save_path + 'GRAY/' + save_name + 'frame%d_opticalflow.bmp' % (i + 1),cur_gray_output)

    return