You will need to have the `gcc` compiler installed. Also, if you are on a Mac, you will need to install automake (can be done with `brew install automake`), as this is required for us to install PhyML for you.

You wil need to `chmod -R 555 test_input_data/` for the tests to pass (our code checks that cached files are in mode 555 before reading them to ensure that they are not corrupted. This is a feature, not a bug.)

Please run the FULL test suit at least twice (i.e. don't use `-x` flag in pytest). The first time tests might fail because some things are getting installed by the test code (such as FastTree, XRATE and PhyML), and the order of execution causes tests to fail. On the bright side, we install all required software for you.
