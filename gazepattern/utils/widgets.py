# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.forms import widgets
from django.template import loader
from django.utils.safestring import mark_safe


class SimpleCustomWidget(widgets.Widget):
    template = None
    request = None
    required = False

    def __init__(self, *args, **kwargs):
        assert (self.template)
        self.request = kwargs.pop("request", False)
        self.required = kwargs.pop('required', False)
        self.label = kwargs.pop('label', False)
        super(SimpleCustomWidget, self).__init__(*args, **kwargs)

    def get_context(self, name, value, attrs=None):

        if not value:
            value = ""

        return {
            'widget': {
                'name': name,
                'value': value,
                "required": self.required,
                "label": self.label,
            }
        }

    def render(self, name, value, attrs=None, renderer=None):
        context = self.get_context(name, value, attrs)
        template = loader.get_template(self.template).render(context)
        return mark_safe(template)


class ChoiceWidget(SimpleCustomWidget):
    template = "utils/widgets/choice_widget.html"
    choices = None

    def __init__(self, *args, **kwargs):
        self.choices = kwargs.pop('choices', False)
        assert(self.choices)
        super(ChoiceWidget, self).__init__(*args, **kwargs)

    def get_context(self, name, value, attrs=None):
        choices_dict = {}
        for choice_item in self.choices:
            choices_dict[choice_item[0]] = choice_item[1]

        required = False
        if attrs:
            required = True if attrs.get('required', False) else False
        
        display = False
        if value:
            value = int(value)
            display = choices_dict.get(int(value))

        return {
            'widget': {
                'name': name,
                'value': value,
                'display': display,
                'choices': choices_dict,
                'required': required,
            }
        }

class TextWidget(SimpleCustomWidget):
    template = "utils/widgets/text_widget.html"