# Nowoczesne Technologie Obliczeniowe - 2019
## Projekt CUDA - propagacja światła za przeźroczem - model 3D

Projekt opierający się o komunikację MPI i bibliotekę CUDA, by wykorzystać pełny potencjał obliczeniowy klastra DWARF wydziału Fizyki PW. <br />
Poniżej jest propozycja listy zadań, które doprowadzą do stworzenia programu projektowego.

* [ ] Przegląd Literatury

Propagacja światła koherentnego przez dowolne przeźrocze - dyfrakcja - jest ciągle ciekawym tematem dla naukowców - pomaga zrozumieć naturę światła i jest szeroko wykorzystywana podczas symulacji układów optycznych. 
Istnieje kilka metod symulacji propagacji światła m.in. liczenie całki tablicy wejściowej oraz metoda splotowa.


* [ ] Wybór wydajnego algorytmu

Jako metodę symulacji propagacji światła wykorzystana zostanie metoda splotowa. Bardzo dobrą literaturą ukazującą algorytm, jak i zalety i wady tej metody jest artykuł Profesora Macieja Sypka [[1]](https://gitlab.com/SimonPW/nto-2019/blob/master/B_01_199504_OptComm.PDF).


* [ ] Analiza i optymalizacja algorytmu
* [ ] Przykładowy kod w Pythonie
* [ ] Implementacja algorytmu w C++
* [ ] Optymalizacja kodu pod kątem CUDA
* [ ] Wykorzystanie MPI i CUDA
* [ ] Docelowa architektura wykorzystująca protokół MPI i wiele GPU na jednym węźle