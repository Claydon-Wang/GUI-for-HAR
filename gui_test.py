import tkinter as tk
from Layer_CNN import CNN
from dataset.data_proc import *

from train_test.train_test_proc import *

def gui():
    test_x_list = "./dataset/WISDM/test_x.npy"
    test_y_list = "./dataset/WISDM/test_y_new.npy"

    top = tk.Tk()

    top.title("WISDM-GUI")
    width = 260
    height = 420
    top.geometry(f'{width}x{height}')

    var_Walking = tk.StringVar()
    var_Upstair = tk.StringVar()
    var_Downstairs = tk.StringVar()
    var_Sitting = tk.StringVar()
    var_Standing = tk.StringVar()
    var_Jogging = tk.StringVar()
    inference_time = tk.StringVar()
    model = tk.StringVar()

    var_Walking.set("?")
    var_Upstair.set("?")
    var_Downstairs.set("?")
    var_Sitting.set("?")
    var_Standing.set("?")
    var_Jogging.set("?")
    inference_time.set("?")
    model.set("?")

    text_Walking = tk.Label(top, text="Walking").place(x=50, y=0)
    text_Upstairs = tk.Label(top, text="Walking Upstairs").place(x=50, y=20)
    text_Downstairs = tk.Label(top, text="Walking Downstairs").place(x=50, y=40)
    text_Sitting = tk.Label(top, text="Sitting").place(x=50, y=60)
    text_Standing = tk.Label(top, text="Standing").place(x=50, y=80)
    text_Jogging = tk.Label(top, text="Jogging").place(x=50, y=100)  # Jogging
    text_time = tk.Label(top, text="Inference time").place(x=50, y=120)
    text_mdoel = tk.Label(top, text="Model").place(x=50, y=140)


    # time_ = tk.Label(top, text="inference time:").place(x=50, y=140)
    # time_2 = tk.Label(top, text="295.213ms").place(x=170, y=140)

    text_Walking_value = tk.Label(top, textvariable=var_Walking).place(x=200, y=0)
    text_Upstairs_value = tk.Label(top, textvariable=var_Upstair).place(x=200, y=20)
    text_Downstairs_value = tk.Label(top, textvariable=var_Downstairs).place(x=200, y=40)
    text_Sitting_value = tk.Label(top, textvariable=var_Sitting).place(x=200, y=60)
    text_Standing_value = tk.Label(top, textvariable=var_Standing).place(x=200, y=80)
    text_Laying_value = tk.Label(top, textvariable=var_Jogging).place(x=200, y=100)
    text_time_value = tk.Label(top, textvariable=inference_time).place(x=200, y=120)
    text_mdoel_value = tk.Label(top, textvariable=model).place(x=200, y=140)


    img_gif = tk.PhotoImage(file='./actions/question.gif')
    img_gif0 = tk.PhotoImage(file='./actions/walking.gif')
    img_gif1 = tk.PhotoImage(file='./actions/upstairs.gif')
    img_gif2 = tk.PhotoImage(file='./actions/downstairs.gif')
    img_gif3 = tk.PhotoImage(file='./actions/sitting.gif')
    img_gif4 = tk.PhotoImage(file='./actions/standing.gif')
    img_gif5 = tk.PhotoImage(file='./actions/jogging.gif')

    label_img = tk.Label(top, image=img_gif)
    label_img.place(x=30, y=160)  # 30  120

    def Clear_result():
        var_Walking.set("?")
        var_Upstair.set("?")
        var_Downstairs.set("?")
        var_Sitting.set("?")
        var_Standing.set("?")
        var_Jogging.set("?")
        label_img.configure(image=img_gif)

    def uci_gui_test():
        global prob
        global result

        data_test = HAR_one_tensor(test_x_list, test_y_list)
        har_test_tensor = data_test.HAR_one_tensor_data()

        test_loader = Data.DataLoader(dataset=har_test_tensor, batch_size=1, shuffle=True, )

        model = torch.load('./model_save/WISDM/WISDM_gMLP_model.pth', map_location='cpu')
        model = model.to(torch.device("cpu"))

        preds_prob, preds = test_one_tensor(model, test_loader)
        prob = np.around(preds_prob.numpy(), decimals=2)[0]
        result = preds

        var_Walking.set(prob[0])
        var_Upstair.set(prob[1])
        var_Downstairs.set(prob[2])
        var_Sitting.set(prob[3])
        var_Standing.set(prob[4])
        var_Jogging.set(prob[5])
        if result == 0:
            label_img.configure(image=img_gif0)
        elif result == 1:
            label_img.configure(image=img_gif1)
        elif result == 2:
            label_img.configure(image=img_gif2)
        elif result == 3:
            label_img.configure(image=img_gif3)
        elif result == 4:
            label_img.configure(image=img_gif4)
        elif result == 5:
            label_img.configure(image=img_gif5)


    button = tk.Button(top, text='Prediction', command=uci_gui_test)
    button.place(x=60, y=370)  # button.place(x=60, y=330)

    button_Clear = tk.Button(top, text='Clear', command=Clear_result)
    button_Clear.place(x=150, y=370)  # button_Clear.place(x=150, y=330)

    top.mainloop()


if __name__ == "__main__":
    gui()



