from app.normalize import normalize, numbers


def test_formatting_differences_collapse():
    assert normalize("  Define the REFLECTION of light!! ") == normalize("define the reflection of light")


def test_degree_sign_kept():
    assert "30°" in normalize("The angle is 30°.")


def test_numbers():
    assert numbers("angle of incidence is 30° and 45.5") == ["30", "45.5"]
    assert numbers("what is the angle of reflection?") == []
