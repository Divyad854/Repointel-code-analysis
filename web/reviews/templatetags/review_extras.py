from django import template

register = template.Library()


@register.filter
def score_class(value):
    if value is None or value == "":
        return "na"
    value = int(value)
    return "good" if value >= 80 else "ok" if value >= 60 else "bad"


@register.filter
def score_text(value):
    return "—" if value is None or value == "" else value


@register.filter
def get_item(mapping, key):
    try:
        return mapping.get(key)
    except AttributeError:
        return None


@register.filter
def pct(value):
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


@register.filter
def pretty_category(value):
    return {"best_practices": "Best practices", "quality": "Code quality"}.get(value, str(value).replace("_", " ").title())
