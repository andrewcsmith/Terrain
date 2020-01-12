from ratio_presenter import RatioPresenter

class StructurePresenter:
    def __init__(self, structure):
        self.structure = structure
    
    def __repr__(self):
        duo_ratios = self.structure.duo.current_ratio() + self.structure.current_base_ratio
        base_ratio = self.structure.current_base_ratio
        duo = f"<{RatioPresenter(duo_ratios[0])} {RatioPresenter(duo_ratios[1])}>"
        base = RatioPresenter(base_ratio)
        return f"{base}: {duo}"
