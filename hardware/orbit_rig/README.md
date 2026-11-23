# Soporte orbital para filmar un tomate con la D435i

El tomate queda fijo en un poste. La cámara gira a su alrededor en dos ejes:
- **Azimut:** un rotor sobre bolitas.
- **Elevación:** un carro sobre un arco de −14° a 80°, siempre apuntando al centro del tomate.

Se mueve la cámara y no el tomate porque `splat_from_video.py` entrena con el fondo, y para COLMAP el fondo tiene que quedar quieto respecto al objeto.

```bash
~/.venvs/cad/bin/python hardware/orbit_rig/orbit_rig.py      # build123d en un venv aparte
```

El script genera en `~/data/cad/orbit_rig/`:
- un STL por pieza, ya orientado y centrado para imprimir;
- `ensamblado.step`;
- `vista_previa.png`.

Las medidas se cambian en los parámetros al inicio del script: la cama, el radio del arco, la altura del tomate y las holguras.

## Piezas (Bambu Lab A1 mini, 180 × 180 × 180 mm)

Todas caben en la cama y ninguna necesita soportes.

```bash
python3 hardware/orbit_rig/slice_bambu.py
```

Lamina con Bambu Studio y su perfil de la A1 mini, y deja en `~/data/cad/orbit_rig/bambu/` un `.3mf` por placa, con los ajustes dentro y listo para abrir e imprimir.

El perfil es "0.28mm Extra Draft" con estos cambios:
- 3 capas arriba y 2 abajo;
- 10 % de relleno;
- 3 paredes en la placa del carro, porque lleva el tornillo que lo aprieta.

| Placa | Piezas | Tiempo | PLA |
|---|---|---|---|
| `1_base` | base calada: pista, 3 rayos y anillo con escala cada 5° y agujeros del pasador cada 15° | 1 h 10 min | 36 g |
| `2_rotor` | rotor con ventanas, copa de contrapeso y puntero | 59 min | 37 g |
| `3_brazo` | brazo (en diagonal) | 33 min | 28 g |
| `4_arcos` | los 4 tramos del arco (radio 230 mm; `arco_1` lleva la pata) | 1 h 06 min | 42 g |
| `5_carro_poste` | carro, poste y punta-pedestal | 46 min | 27 g |
| total | | **4 h 33 min** | **170 g** |
| `opcional_colgante` | pinza para colgar el tomate de la rama; se ve desde algunos ángulos | 19 min | 10 g |

El tiempo incluye el arranque de cada placa (calentar, nivelar y purgar), unos 8 min por placa. Por eso las piezas chicas van juntas.

La holgura de las piezas que deslizan es `FIT = 0.2` mm. Si el carro queda apretado, sube `FIT` en `orbit_rig.py`, regenera y vuelve a laminar.

Para instalar Bambu Studio sin sudo:
1. Bajar el AppImage de GitHub y extraerlo con `--appimage-extract` en `~/.local/opt/bambu`.
2. Ubuntu 24.04 no trae WebKitGTK 4.1, así que hay que bajar estos paquetes con `apt-get download` y extraerlos con `dpkg-deb -x` en `~/.local/opt/bambu/libs`:
   - `libwebkit2gtk-4.1-0`
   - `libjavascriptcoregtk-4.1-0`
   - `libmanette-0.2-0`
   - `libenchant-2-2`
   - `libhyphen0`
   - `libsecret-1-0`

## Materiales extra

- ~60 bolitas de 6 mm (airsoft o acero) para la pista entre base y rotor
- 4 tornillos M4×10 con tuerca (brazo → rotor)
- 1 tornillo M4×16 con tuerca (prisionero del carro)
- 6 tornillos M3×12 con tuerca (uniones del arco)
- 1 tornillo M3×20 con tuerca (pata → brazo)
- 1 tornillo M3×10 (prisionero de la punta, autorroscante en el plástico)
- 1 tornillo de cámara 1/4"-20 de unos 10 mm, o el de un trípode
- 1 pasador de 3 mm (un clavo o un tornillo M3) para fijar el azimut
- 150–200 g de contrapeso en la copa (tuercas, perdigones). Sin contrapeso la base no se vuelca, pero queda justa.

## Armado

1. Pegar el poste a la base (CA o epoxi).
2. Poner las bolitas en la ranura de la base y bajar el rotor.
3. Insertar el poste: su ala impide que el rotor se levante.
4. Poner las tuercas M4 en los hexágonos del rotor y atornillar el brazo.
5. Meter el carro en `arco_4` por abajo.
6. Unir los tramos de arriba hacia abajo. Cada lengüeta entra en la ranura del tramo siguiente y lleva dos M3, con la cabeza en la cara lisa y la tuerca en la grabada.
7. Calzar la pata en el brazo con el M3×20.
8. Atornillar la cámara a la placa del carro.
9. Llenar la copa de contrapeso.

## Captura

- **Distancia:** el frente de la cámara queda a 146–196 mm del tomate, según la posición en la ranura. El centro del fruto queda a unos 122 mm de la mesa. Con la copa, se ajusta subiendo la punta (prisionero M3).
- **Pasadas:** varias vueltas completas a elevación fija, por ejemplo −10°, 10°, 30°, 50° y 70°. La elevación se lee en la ranura del carro. Para fotos, los agujeros cada 15° dan 24 posiciones por vuelta.
- **Colores:** imprimir rotor, brazo y arco en negro mate, para que lo que se mueve con la cámara no le dé puntos a COLMAP. La base sí conviene que tenga textura, por ejemplo un papel impreso pegado, para dar referencias fijas.
- **Cable:** el USB da una vuelta al poste por cada vuelta del brazo. Conviene alternar el sentido de giro entre pasadas.
- **Pedestal en el splat:** con la punta-pedestal, el poste queda pegado al fruto y `real_tomato_object.py` lo incluiría en el objeto. Hay que descartar lo que queda bajo la base del fruto antes de usarlo.
