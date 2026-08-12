from sg_rife.IFNet_dino import IFNet


def test_ifnet_checkpoint_structure():
    model = IFNet(dino_in_channels=384, dino_patch_size=16)
    state = model.state_dict()

    assert state["block0.conv0.0.0.weight"].shape[1] == 6
    assert state["dino_compressor.shared_basis.weight"].shape[1] == 384
    assert "unet.fusion_s3.dcn_weight" in state
