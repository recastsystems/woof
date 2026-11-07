import pytest
from woof import cli

@pytest.mark.parametrize("door",["--wrfinput","--met-em"])
def test_skin_product_request_is_accepted_by_real_cli_parser(door,tmp_path):
    args=cli.build_parser().parse_args(["run",door,str(tmp_path),"--products","var:wrf_sstsk,var:wrf_dtw"])
    assert args.render_products=="var:wrf_sstsk,var:wrf_dtw"

def test_config_route_refuses_an_unconsumed_product_request(tmp_path,capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["run",str(tmp_path/"config.toml"),"--products","none"])
    assert error.value.code==2
    assert "CONFIG run route does not consume" in capsys.readouterr().err

def test_metem_worker_initializes_all_helpers_before_entering_cli():
    import ast
    from pathlib import Path
    from woof import metem_forecast
    tree=ast.parse(Path(metem_forecast.__file__).read_text())
    entry=[node for node in tree.body if isinstance(node,ast.If) and "__name__" in ast.unparse(node.test)]
    assert len(entry)==1
    assert all(node.lineno < entry[0].lineno for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))
