import serial

class SerialWriter:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate)

    def __del__(self):
        self.ser.close()

    def write(self, yaw_angle, pitch_angle):
        """
        Protocol: 
        0xFA + length(2 bytes) + yaw_angle(2 bytes) + pitch_angle(2 bytes) + 0x00(reserved) + 0xFB
        """
        mes = bytearray(
            [0xFA, 0x04, 0x00,
             yaw_angle & 0xFF, (yaw_angle >> 8) & 0xFF,
             pitch_angle & 0xFF, (pitch_angle >> 8) & 0xFF,
             0x00, 0xFB])
        self.ser.write(mes)

    # writer = SerialWriter('/dev/ttyUSB0', 115200)
    # writer.write(5000, 5000)
    # cap = cv2.VideoCapture(0)

