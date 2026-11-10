from nowcast.backtest import is_rain


def test_metar_rain_codes():
    for wx in ["-RA", "+SHRA", "-SHRA BR", "TSRA", "-RADZ", "-DZ BR", "SHRA VCTS", "-TSRA"]:
        assert is_rain(wx), wx
    for wx in ["VCSH", "VCTS", "BR", "HZ", "M", ""]:
        assert not is_rain(wx), wx
