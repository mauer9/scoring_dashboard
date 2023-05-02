from docxtpl import DocxTemplate, InlineImage


def get_context():
    return {
        'invoice_no': 12345,
        'date': '30 Mar',
        'due_date': '30 Apr',
        'name': 'Jane Doe',
        'address': '123 Quiet Lane',
        'subtotal': 335,
        'tax_amt': 10,
        'total': 345,
        'amt_paid': 100,
        'amt_due': 245,
        'row_contents': [
            {
                'description': 'Eggs',
                'quantity': 30,
                'rate': 5,
                'amount': 150
            }, {
                'description': 'All Purpose Flour',
                'quantity': 10,
                'rate': 15,
                'amount': 150
            }, {
                'description': 'Eggs',
                'quantity': 5,
                'rate': 7,
                'amount': 35
            }
        ]
    }


doc = DocxTemplate('monitoring_report_template.docx')
context = get_context()

doc.render(context)
doc.save("generated_doc.docx")