# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import pandas as pd
import pytest

import secretflow.compute as sc
from secretflow.component.data_utils import DistDataType, VerticalTableWrapper
from secretflow.component.preprocessing.unified_single_party_ops.cast import (
    CAST_VERSION,
    _apply_rules_on_table,
    cast_comp,
)
from secretflow.component.preprocessing.unified_single_party_ops.substitution import (
    substitution,
)
from secretflow.component.storage.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def test_cast(comp_prod_sf_cluster_config):
    alice_input_path = "test_cast/alice_input.csv"
    bob_input_path = "test_cast/bob_input.csv"
    output_path = "test_cast/output.csv"
    output_rule = "test_cast/output.rule"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    alice_data = {"A": ["s1", "s2"], "B": [1, 2], "C": ["0.1", "0.2"]}
    bob_data = {"D": [0.1, 1.2], "E": [True, False]}

    expected_alice = {"A": ["s1", "s2"], "B": [1.0, 2.0], "C": [0.1, 0.2]}
    expected_bob = {"D": [0.1, 1.2], "E": [1.0, 0.0]}

    if self_party == "alice":
        pd.DataFrame(alice_data).to_csv(
            comp_storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        pd.DataFrame(bob_data).to_csv(
            comp_storage.get_writer(bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="preprocessing",
        name="cast",
        version=CAST_VERSION,
        attr_paths=[
            "astype",
            "input/input_ds/columns",
        ],
        attrs=[Attribute(s="float"), Attribute(ss=["B", "C", "D", "E"])],
        inputs=[
            DistData(
                name="input_ds",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            )
        ],
        output_uris=[output_path, output_rule],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["str", "int", "str"],
                features=["A", "B", "C"],
            ),
            TableSchema(
                feature_types=["float"],
                features=["D"],
                label_types=["bool"],
                labels=["E"],
            ),
        ]
    )
    param.inputs[0].meta.Pack(meta)
    res = cast_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    assert len(res.outputs) == 2

    if self_party in ["alice", "bob"]:
        out_meta = VerticalTable()
        res.outputs[0].meta.Unpack(out_meta)
        logging.warning(f"...meta.... \n{out_meta}\n.......")
        assert out_meta.schemas[0].feature_types == ["str", "float64", "float64"]
        assert out_meta.schemas[1].feature_types == ["float"]
        assert out_meta.schemas[1].label_types == ["float64"]

        comp_storage = ComponentStorage(storage_config)
        real_data = pd.read_csv(comp_storage.get_reader(output_path))
        logging.warning(f"...vertical_res:{self_party}... \n{real_data}\n.....")
        expected_data = expected_alice if self_party == "alice" else expected_bob
        pd.testing.assert_frame_equal(
            pd.DataFrame(expected_data),
            real_data,
        )
    # test substitution
    sub_path = "test_cast/substitution.csv"
    param2 = NodeEvalParam(
        domain="preprocessing",
        name="substitution",
        version="0.0.2",
        inputs=[param.inputs[0], res.outputs[1]],
        output_uris=[sub_path],
    )
    res = substitution.eval(
        param=param2,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    if self_party in ["alice", "bob"]:
        comp_storage = ComponentStorage(storage_config)
        sub_out = pd.read_csv(comp_storage.get_reader(sub_path))
        original_out = pd.read_csv(comp_storage.get_reader(output_path))

        logging.warning(f"....... \nsub_out\n{sub_out}\n.,......")

        assert sub_out.equals(original_out)

    # test raise error
    logging.warning("test raise error")
    param.attrs.pop()
    param.attrs.append(Attribute(ss=["A"]))

    with pytest.raises(ValueError) as exc_info:
        cast_comp.eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )
    logging.warning(f"Caught expected Exception: {exc_info}")


def test_cast_apply_table():
    good_cases = [
        {"data": ["\' 1\' ", "\"2\"", " 3 "], "except": [1, 2, 3], "target": "int"},
        {
            "data": ["\' 1\' ", "\"2\"", " 3.0 "],
            "except": [1.0, 2.0, 3.0],
            "target": "float",
        },
        {
            "data": [" 1 ", "2", " 3.0 "],
            "except": [" 1 ", "2", " 3.0 "],
            "target": "str",
        },
        {"data": [1, 2, 3], "except": [1, 2, 3], "target": "int"},
        {"data": [1, 2, 3], "except": [1.0, 2.0, 3.0], "target": "float"},
        {"data": [1, 2, 3], "except": ["1", "2", "3"], "target": "str"},
        {"data": [1.0, 2.0, 3.0], "except": [1, 2, 3], "target": "int"},
        {"data": [1.0, 2.0, 3.0], "except": [1.0, 2.0, 3.0], "target": "float"},
        {"data": [1.0, 2.0, 3.0], "except": ["1", "2", "3"], "target": "str"},
        {"data": [True, False, True], "except": [1, 0, 1], "target": "int"},
        {"data": [True, False, True], "except": [1.0, 0.0, 1.0], "target": "float"},
        {
            "data": [True, False, True],
            "except": ["true", "false", "true"],
            "target": "str",
        },
    ]

    for index, item in enumerate(good_cases):
        df = pd.DataFrame({"A": item["data"]})
        table = sc.Table.from_pandas(df)
        res_tbl = _apply_rules_on_table(table, item["target"])
        res = res_tbl.to_pandas()["A"].tolist()
        excepted = item["except"]
        assert res == excepted, f"not equal: {index}, {res}, {excepted}"

    bad_cases = [
        {"data": ["1.0"], "target": "int"},
        {"data": [""], "target": "int"},
        {"data": [""], "target": "float"},
        {"data": ["a"], "target": "float"},
        {"data": ["a"], "target": "int"},
    ]
    for index, item in enumerate(bad_cases):
        with pytest.raises(ValueError) as exc_info:
            df = pd.DataFrame({"A": item["data"]})
            table = sc.Table.from_pandas(df)
            _apply_rules_on_table(table, item["target"])
        logging.warning(f"Caught expected Exception: {index},{exc_info}")
