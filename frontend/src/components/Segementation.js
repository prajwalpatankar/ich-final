import React, { useEffect, useState } from 'react';
import { SpinnerInfinity } from 'spinners-react';
// import listReactFiles from 'list-react-files'


function ResResultsSegementedults({ output, output_masked }) {

    const [r, setR] = useState(false);
    const [imageState, setImageState] = useState(false);

    const classes = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "subdural"]

    useEffect(() => {
        setTimeout(() => {
            setR(true)
        }, 100)

        setTimeout(() => {
            setImageState(true)
        }, 2000)

        // listReactFiles(process.env.PUBLIC_URL + "converted_ct").then(files => console.log("files :",files))

    })

    return (
        <div>
            <p>Results</p>
            {r ?
                <div>
                    <br />
                    <p>Segmented Output : </p>
                    {imageState ?
                        <div>
                            <img src={output} alt="brain" /> &nbsp;
                            <img className="origImage" src={output} alt="brain" />
                            <img className="mask" src={output_masked} alt="brain" />
                        </div>
                        :
                        <SpinnerInfinity />
                    }
                </div>

                :
                <p> </p>
            }
        </div>
    )

}

export default ResultsSegemented;