import pandas as pd
import gradio as gr

# wczytanie pliku
def load_file(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print("Nie znaleziono pliku!")

# wypisanie części tabeli oraz informacji o tabeli
def print_table_info(data, num_rows):
    output_text = "\nCzęść tabeli:\n\n" + str(data.head(num_rows)) + "\n\nInformacje o tabeli:\n" + \
                  "Liczba obiektów: " + str(len(data)) + "\nLiczba atrybutów: " + str(len(data.columns)) + \
                  "\nNazwy atrybutów: " + str(list(data.columns))
    return output_text

# obliczanie liczby klas decyzyjnych
def count_classes(data):
    return len(data.iloc[:, -1].unique())

# obliczanie wielkości każdej klasy decyzyjnej
def count_class_sizes(data):
    class_sizes = data.iloc[:, -1].value_counts()
    return dict(class_sizes)

# chatbot Gradio
def chatbot(filename, num_rows):
    # ładowanie pliku
    data = load_file(filename)

    if data is not None:
        # wypisanie części tabeli oraz informacji o tabeli
        output_text = print_table_info(data, int(num_rows))

        # dodatkowe pytania dotyczące tabeli
        choice = gr.inputs.Radio(["Tak", "Nie"], default="Nie", label="Czy chcesz zadać dodatkowe pytania o tabeli decyzyjnej?")
        if choice == "Tak":
            output_text += "\n\nLiczba klas decyzyjnych: " + str(count_classes(data)) + \
                           "\nWielkość każdej klasy decyzyjnej: " + str(count_class_sizes(data))

        return output_text

# interfejs Gradio
iface = gr.Interface(fn=chatbot,
                     inputs=["text", gr.inputs.Number(default=10)],
                     outputs="text",
                     title="Chatbot Tabeli Decyzyjnej",
                     description="Wprowadź nazwę pliku i liczbę wierszy, aby wyświetlić część tabeli oraz informacje o tabeli decyzyjnej.")

iface.launch()
