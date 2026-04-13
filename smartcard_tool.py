from smartcard.System import readers

def send_apdu_command(apdu: list) -> dict:
    """
    发送 APDU 命令到第一个可用读卡器，返回响应数据和状态字节。

    :param apdu: 要发送的 APDU 命令（列表形式）
    :return: dict，包含响应数据、十六进制、状态字节等
    """
    reader_list = readers()
    if not reader_list:
        raise RuntimeError("没有检测到读卡器")

    reader = reader_list[0]
    #print(f"使用读卡器: {reader}")

    connection = reader.createConnection()
    connection.connect()

    try:
        # 发送主命令
        response, sw1, sw2 = connection.transmit(apdu)
        #print(f"响应数据: {response}")
        #print(f"状态字节: SW1={hex(sw1)}, SW2={hex(sw2)}")

        full_response = response.copy()

        # 如果有后续数据
        if sw1 == 0x61:
            get_response = [0x00, 0xC0, 0x00, 0x00, sw2]
            response2, sw1_2, sw2_2 = connection.transmit(get_response)
            #print("后续数据:", response2)
            #print("状态字节:", hex(sw1_2), hex(sw2_2))
            full_response.extend(response2)
            sw1, sw2 = sw1_2, sw2_2

        # 转为十六进制字符串
        hex_str = ''.join(f'{b:02X}' for b in full_response)
        #print("十六进制表示:", hex_str)

        return hex_str

    finally:
        try:
            connection.disconnect()
        except:
            pass


# 示例 APDU 命令 与 测试
apdu = [
    0x00, 0x11, 0x00, 0x00, 0x20,
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00
]

result = send_apdu_command(apdu)

print("完整响应（十六进制）:", result)



