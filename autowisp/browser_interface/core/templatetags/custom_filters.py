from django import template

register = template.Library()

@register.filter('get_key')
def get_key(data, key):
    """
    usage example {{ your_dict|get_value_from_dict:your_key }}
    """

    if key:
        try:
            return data.get(key)
        except AttributeError:
            return getattr(data, key)
