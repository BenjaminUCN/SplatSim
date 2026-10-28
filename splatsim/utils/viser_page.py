"""Detalles de página para los paneles viser (robot, revisor de grabaciones, cola de entrenamiento).

Viser sirve siempre el mismo index.html con <title>Viser</title> y no tiene API para cambiarlo, y
add_html inserta el HTML sin ejecutar <script>: el título se pone con el onerror de una imagen
oculta, que sí corre. Así cada panel tiene su propio nombre de pestaña en el navegador.
"""
import html
import json


def set_page_title(server, title: str) -> None:
    """Título de la pestaña del navegador para quien abra este servidor viser."""
    js = f"document.title={json.dumps(title)}"
    server.gui.add_html(f'<img src="data:," alt="" style="display:none" onerror="{html.escape(js)}">')


def link_html(port: int, text: str) -> str:
    """Enlace a otro servidor de esta máquina, en una pestaña nueva; usa el host con que se abrió
    este panel (no localhost), así funciona también desde otro computador."""
    js = f"window.open(location.protocol+'//'+location.hostname+':{int(port)}','_blank');return false"
    return f'<a href="#" onclick="{html.escape(js)}">{html.escape(text)}</a>'
