# Placa para correr la D415 de la muñeca

Con la D415 sobre la cuña de 15°, el punto de corte de la pinza B3 aparece en la esquina derecha de `wrist_rgb`, a 0.76 del ancho. Pasa porque la lente RGB está a 35.2 mm del centro de la carcasa, y la carcasa está centrada en la brida. Así la lente queda en y = −35 mm de `link_eef`, mientras la hoja corta en y = +13.8 mm.

Esta placa se atornilla en los dos M3 de la cuña, que están a ±22.5 mm. Lleva la D415 **invertida** (girada 180° sobre su eje óptico), con la carcasa corrida 20 mm. Con eso la lente queda en y = +15.2 mm, sobre el plano de la hoja, y el punto de corte queda centrado, a 0.49 del ancho.

```bash
~/.venvs/cad/bin/python hardware/camera_shift/camera_shift_plate.py          # --shift -20 por defecto
```

El script deja en `~/data/cad/camera_shift/`:
- `placa_d415_m20.stl`, ya orientada para imprimir y sin soportes;
- `placa_d415_m20.step`;
- `placa_d415_m20.png`, una vista desde la cámara.

Para laminarla para la A1 mini, usa `slice_bambu.py` de `hardware/orbit_rig` con el perfil Extra Draft, 4 paredes y 30 % de relleno. Imprime en **11 min con 7.2 g**.

## Montaje

1. Pon tuercas M3 en los dos hexágonos de arriba.
2. Atornilla la D415 invertida a la placa con 2 × M3 × 6 desde abajo. Las cabezas van en los rebajes, y la lente RGB va hacia la muesca en V.
3. Atornilla la placa a la cuña con los M3 de siempre, desde abajo del soporte. Tienen que ser 5 mm más largos que los de ahora.
4. En el nodo de la cámara, gira la imagen 180°.

## En la simulación

El URDF equivalente sale de:

```bash
build_hand_held_urdf.py ... --camera-wedge-deg 15 --camera-shift-mm 50.4 0 5
```

Ese URDF es `xarm5_b3_d415_w15_s50.urdf`, y la escena es `xarm5_pruner_w15s50`. El 50.4 mm es lo que se corre la lente a la derecha de la imagen, y el 5 mm es el espesor de la placa, a lo largo del eje óptico.

## Pendiente de medir en la pieza real

- **Agujeros de la D415:** verificar con pie de metro que los M3 de la espalda estén a 45 mm y centrados en la carcasa.
- **Rosca de la cámara:** verificar que el M3 × 6 no entre más de lo que permite la rosca de la cámara. Con la placa entran unos 4 mm.
- **Cable USB-C:** al invertir la cámara, el cable sale por el otro extremo. Revisar que no choque con la B3.
