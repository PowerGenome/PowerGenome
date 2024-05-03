The CSV file in this directory allows a user to add their own technology costs and
heat rate assumptions to cases.

technology: Name of the technology. Should be distinct in some way from the ATB names,
especially the ATB <technology>_<tech_detail> combination.

planning_year: The model planning year. This might be different from the basis year that technology costs and attributes are based on.

capex_mw: USD/MW capital expenses, including construction financing costs, etc. Be sure to
convert from $/kW.

capex_mwh: USD/MWh capital expenses, including construction financing costs, etc. Used
for storage resources such as batteries.

fixed_o_m_mw: Fixed O&M costs per MW-year.

fixed_o_m_mwh: Fixed O&M costs per MWh-year. Used for storage resources such as batteries.

variable_o_m_mwh: Variable O&M costs per MWh.

wacc_real: Nominal weighted average cost of capital.

dollar_year: Year that dollars are calculated for. This is used to convert costs to a
consistent dollar year across inputs.

heat_rate: MMBTU per MWh

notes: Not used in calculations, this is a place to document sources, etc.
