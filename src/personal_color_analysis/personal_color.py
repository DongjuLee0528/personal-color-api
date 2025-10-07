from analyzers.personal_color import PersonalColorAnalyzer


def analysis(imgpath):
    """Legacy CLI helper that prints analysis results for ``imgpath``."""

    analyzer = PersonalColorAnalyzer()
    result = analyzer.analyze(imgpath)

    print('Lab_b[skin, eyebrow, eye]', result.lab_b)
    print('hsv_s[skin, eyebrow, eye]', result.hsv_s)
    print('{}의 퍼스널 컬러는 {}입니다.'.format(imgpath, result.tone_label))

    return result
