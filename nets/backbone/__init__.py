from . import darknet

backbone_fn = {
    "darknet_21": darknet.darknet21,
    "darknet_53": darknet.darknet53,
}
# backbone_fn_cc = {
#     "darknet_21": darknet_cc.darknet21,
#     "darknet_53": darknet_cc.darknet53,
# }
# backbone_fn_gn = {
#     "darknet_21": darknet_gn.darknet21,
#     "darknet_53": darknet_gn.darknet53,
# }