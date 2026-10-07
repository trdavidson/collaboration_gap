from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Optional, Union
import fire
import os


TEMPLATE_FOLDER = './data/prompt_templates'


def render_template(
        template_name: str,
        user_data: Optional[dict] = None,
        template_folder: Optional[str] = None
) -> str:
    """
    Render a Jinja2 template with user data and automatic whitespace control.

    Args:
        template_name: Name of the template file (e.g., 'game_mode_collab.j2')
        user_data: Dictionary of variables to pass to the template
        template_folder: Optional folder path where templates are stored.
                         Defaults to TEMPLATE_FOLDER if not provided.

    Returns:
        Rendered template as string
    """

    if user_data is None:
        user_data = {}

    if template_folder is None:
        template_folder = TEMPLATE_FOLDER

    # Ensure template_dirs is a list
    template_dirs = [os.path.join(template_folder, f)
                     for f in os.listdir(template_folder)]

    # Create environment with whitespace control
    env = Environment(
        loader=FileSystemLoader(template_dirs),
        autoescape=select_autoescape(['html', 'xml']),
        # Automatic whitespace trimming
        trim_blocks=True,      # Remove newline after a block
        lstrip_blocks=True,    # Remove spaces/tabs at beginning of line
        keep_trailing_newline=False  # Don't keep final newline
    )

    # Load and render template
    template = env.get_template(template_name)
    rendered = template.render(**user_data)

    return rendered


def main(template_name: str, user_data: Optional[dict] = None):

    if user_data is None:
        user_data = {
            "hidden_coordinates": True,
            "map_str": "<some map>"
        }

    # Example usage
    template = render_template(
        template_name=template_name,
        user_data=user_data
    )
    print(template)


if __name__ == "__main__":
    fire.Fire(main)
