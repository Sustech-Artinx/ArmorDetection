import serial


class SerialWriter:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate)

    def __del__(self):
        self.ser.close()

    @staticmethod
    def _to_int16(n):
        if n < 0:
            return 65536 + n
        else:
            return n

    def write(self, delta_yaw, delta_pitch):
        """
        Protocol:
        0xFA + length(2 bytes) + d_yaw(2 bytes) + d_pitch(2 bytes) + 0x00(reserved) + 0xFB
        """
        delta_yaw = SerialWriter._to_int16(delta_yaw)
        delta_pitch = SerialWriter._to_int16(delta_pitch)

        mes = bytearray(
            [0xFA, 0x04, 0x00,
             delta_yaw & 0xFF, (delta_yaw >> 8) & 0xFF,
             delta_pitch & 0xFF, (delta_pitch >> 8) & 0xFF,
             0x00, 0xFB])
        self.ser.write(mes)


if __name__ == '__main__':
    writer = SerialWriter('/dev/ttyUSB0', 115200)
    writer.write(0, 0)